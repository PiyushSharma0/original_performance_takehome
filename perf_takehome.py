"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(self, *engine_slots):
        bundle = defaultdict(list)
        for engine, slot in engine_slots:
            bundle[engine].append(slot)
        self.instrs.append(dict(bundle))

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def _vec_const(self, scalar_const_addr, name):
        addr = self.alloc_scratch(name, VLEN)
        self.add("valu", ("vbroadcast", addr, scalar_const_addr))
        return addr

    def build_hash_vec(self, val_hash_addr, tmp1, tmp2, consts):
        (
            c_mul_0,
            c_add_0,
            c_add_1,
            c_mul_1,
            c_add_2,
            c_add_3,
            c_mul_2,
            c_add_4,
            c_add_5,
            c_shift_9,
            c_shift_16,
            c_shift_19,
        ) = consts

        # stage 0: (x + C) + ((x + C) << 12) == (x + C) * 4097
        self.add("valu", ("multiply_add", val_hash_addr, val_hash_addr, c_mul_0, c_add_0))
        # stage 1
        self.add("valu", ("^", tmp1, val_hash_addr, c_add_1))
        self.add("valu", (">>", tmp2, val_hash_addr, c_shift_19))
        self.add("valu", ("^", val_hash_addr, tmp1, tmp2))
        # stage 2: (x + C) + ((x + C) << 5) == (x + C) * 33
        self.add("valu", ("multiply_add", val_hash_addr, val_hash_addr, c_mul_1, c_add_2))
        # stage 3
        self.add("valu", ("+", tmp1, val_hash_addr, c_add_3))
        self.add("valu", ("<<", tmp2, val_hash_addr, c_shift_9))
        self.add("valu", ("^", val_hash_addr, tmp1, tmp2))
        # stage 4: (x + C) + ((x + C) << 3) == (x + C) * 9
        self.add("valu", ("multiply_add", val_hash_addr, val_hash_addr, c_mul_2, c_add_4))
        # stage 5
        self.add("valu", ("^", tmp1, val_hash_addr, c_add_5))
        self.add("valu", (">>", tmp2, val_hash_addr, c_shift_16))
        self.add("valu", ("^", val_hash_addr, tmp1, tmp2))

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation that keeps the working values in scratch and
        processes the batch in 8-wide chunks.
        """
        # Working arrays live in scratch so the hot loop avoids repeated
        # load/store traffic.
        vals = self.alloc_scratch("vals", batch_size)
        idxs = self.alloc_scratch("idxs", batch_size)

        # Scalar temporaries.
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")

        # Vector temporaries. We keep several chunks in flight so the valu
        # engine can be filled across independent chunks in the same round.
        pack_width = 6
        node_addr = [
            self.alloc_scratch(f"node_addr_{i}", VLEN) for i in range(pack_width)
        ]
        node_vals = [
            self.alloc_scratch(f"node_vals_{i}", VLEN) for i in range(pack_width)
        ]
        tmp1 = [self.alloc_scratch(f"tmp1_{i}", VLEN) for i in range(pack_width)]
        tmp2 = [self.alloc_scratch(f"tmp2_{i}", VLEN) for i in range(pack_width)]
        tmp3 = [self.alloc_scratch(f"tmp3_{i}", VLEN) for i in range(pack_width)]

        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_addr, i))
            self.add("load", ("load", self.scratch[v], tmp_addr))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        vec_zero = self._vec_const(zero_const, "vec_zero")
        vec_one = self._vec_const(one_const, "vec_one")
        vec_two = self._vec_const(two_const, "vec_two")

        # Vector constants used in the hash.
        c_add_0 = self._vec_const(self.scratch_const(0x7ED55D16), "c_add_0")
        c_add_1 = self._vec_const(self.scratch_const(0xC761C23C), "c_add_1")
        c_add_2 = self._vec_const(self.scratch_const(0x165667B1), "c_add_2")
        c_add_3 = self._vec_const(self.scratch_const(0xD3A2646C), "c_add_3")
        c_add_4 = self._vec_const(self.scratch_const(0xFD7046C5), "c_add_4")
        c_add_5 = self._vec_const(self.scratch_const(0xB55A4F09), "c_add_5")
        c_mul_0 = self._vec_const(self.scratch_const(4097), "c_mul_0")
        c_mul_1 = self._vec_const(self.scratch_const(33), "c_mul_1")
        c_mul_2 = self._vec_const(self.scratch_const(9), "c_mul_2")
        c_shift_9 = self._vec_const(self.scratch_const(9), "c_shift_9")
        c_shift_16 = self._vec_const(self.scratch_const(16), "c_shift_16")
        c_shift_19 = self._vec_const(self.scratch_const(19), "c_shift_19")
        c_n_nodes = self._vec_const(self.scratch["n_nodes"], "c_n_nodes")
        c_forest_base = self._vec_const(self.scratch["forest_values_p"], "c_forest_base")

        hash_consts = (
            c_mul_0,
            c_add_0,
            c_add_1,
            c_mul_1,
            c_add_2,
            c_add_3,
            c_mul_2,
            c_add_4,
            c_add_5,
            c_shift_9,
            c_shift_16,
            c_shift_19,
        )

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))

        # Load the initial values into scratch.
        for offset in range(0, batch_size, 2 * VLEN):
            off = self.scratch_const(offset)
            self.add("alu", ("+", tmp_addr, self.scratch["inp_values_p"], off))
            if offset + VLEN < batch_size:
                off2 = self.scratch_const(offset + VLEN)
                self.add("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], off2))
                self.add_bundle(
                    ("load", ("vload", vals + offset, tmp_addr)),
                    ("load", ("vload", vals + offset + VLEN, tmp_addr2)),
                )
            else:
                self.add("load", ("vload", vals + offset, tmp_addr))

        # The indices start at zero for every sample.
        # Scratch is zero-initialized, so we can keep idxs as-is.

        # Process the rounds in chunk groups so independent chunks can be
        # bundled together across the 6-slot valu engine.
        for _round in range(rounds):
            for base in range(0, batch_size, pack_width * VLEN):
                active = min(pack_width, (batch_size - base) // VLEN)

                # Compute node addresses for all active chunks.
                self.instrs.append(
                    {
                        "valu": [
                            ("+", node_addr[chunk], c_forest_base, idxs + base + chunk * VLEN)
                            for chunk in range(active)
                        ]
                    }
                )

                # Load node values for every active chunk. Two scalar loads fit in
                # each bundle, so we pair lane offsets.
                for chunk in range(active):
                    for lane in range(0, VLEN, 2):
                        self.add_bundle(
                            (
                                "load",
                                ("load_offset", node_vals[chunk], node_addr[chunk], lane),
                            ),
                            (
                                "load",
                                ("load_offset", node_vals[chunk], node_addr[chunk], lane + 1),
                            ),
                        )

                # XOR the node values into the working values.
                self.instrs.append(
                    {
                        "valu": [
                            (
                                "^",
                                vals + base + chunk * VLEN,
                                vals + base + chunk * VLEN,
                                node_vals[chunk],
                            )
                            for chunk in range(active)
                        ]
                    }
                )

                # Hash stage 0.
                self.instrs.append(
                    {
                        "valu": [
                            (
                                "multiply_add",
                                vals + base + chunk * VLEN,
                                vals + base + chunk * VLEN,
                                c_mul_0,
                                c_add_0,
                            )
                            for chunk in range(active)
                        ]
                    }
                )

                # Hash stage 1.
                self.instrs.append(
                    {
                        "valu": [
                            ("^", tmp1[chunk], vals + base + chunk * VLEN, c_add_1)
                            for chunk in range(active)
                        ]
                    }
                )
                self.instrs.append(
                    {
                        "valu": [
                            (">>", tmp2[chunk], vals + base + chunk * VLEN, c_shift_19)
                            for chunk in range(active)
                        ]
                    }
                )
                self.instrs.append(
                    {
                        "valu": [
                            ("^", vals + base + chunk * VLEN, tmp1[chunk], tmp2[chunk])
                            for chunk in range(active)
                        ]
                    }
                )

                # Hash stage 2.
                self.instrs.append(
                    {
                        "valu": [
                            (
                                "multiply_add",
                                vals + base + chunk * VLEN,
                                vals + base + chunk * VLEN,
                                c_mul_1,
                                c_add_2,
                            )
                            for chunk in range(active)
                        ]
                    }
                )

                # Hash stage 3.
                self.instrs.append(
                    {
                        "valu": [
                            ("+", tmp1[chunk], vals + base + chunk * VLEN, c_add_3)
                            for chunk in range(active)
                        ]
                    }
                )
                self.instrs.append(
                    {
                        "valu": [
                            ("<<", tmp2[chunk], vals + base + chunk * VLEN, c_shift_9)
                            for chunk in range(active)
                        ]
                    }
                )
                self.instrs.append(
                    {
                        "valu": [
                            ("^", vals + base + chunk * VLEN, tmp1[chunk], tmp2[chunk])
                            for chunk in range(active)
                        ]
                    }
                )

                # Hash stage 4.
                self.instrs.append(
                    {
                        "valu": [
                            (
                                "multiply_add",
                                vals + base + chunk * VLEN,
                                vals + base + chunk * VLEN,
                                c_mul_2,
                                c_add_4,
                            )
                            for chunk in range(active)
                        ]
                    }
                )

                # Hash stage 5.
                self.instrs.append(
                    {
                        "valu": [
                            ("^", tmp1[chunk], vals + base + chunk * VLEN, c_add_5)
                            for chunk in range(active)
                        ]
                    }
                )
                self.instrs.append(
                    {
                        "valu": [
                            (">>", tmp2[chunk], vals + base + chunk * VLEN, c_shift_16)
                            for chunk in range(active)
                        ]
                    }
                )
                self.instrs.append(
                    {
                        "valu": [
                            ("^", vals + base + chunk * VLEN, tmp1[chunk], tmp2[chunk])
                            for chunk in range(active)
                        ]
                    }
                )

                # Index update.
                self.instrs.append(
                    {
                        "valu": [
                            ("&", tmp1[chunk], vals + base + chunk * VLEN, vec_one)
                            for chunk in range(active)
                        ]
                    }
                )
                self.instrs.append(
                    {
                        "valu": [
                            ("+", tmp3[chunk], tmp1[chunk], vec_one)
                            for chunk in range(active)
                        ]
                    }
                )
                self.instrs.append(
                    {
                        "valu": [
                            (
                                "multiply_add",
                                idxs + base + chunk * VLEN,
                                idxs + base + chunk * VLEN,
                                vec_two,
                                tmp3[chunk],
                            )
                            for chunk in range(active)
                        ]
                    }
                )
                self.instrs.append(
                    {
                        "valu": [
                            ("<", tmp1[chunk], idxs + base + chunk * VLEN, c_n_nodes)
                            for chunk in range(active)
                        ]
                    }
                )
                self.instrs.append(
                    {
                        "valu": [
                            (
                                "*",
                                idxs + base + chunk * VLEN,
                                idxs + base + chunk * VLEN,
                                tmp1[chunk],
                            )
                            for chunk in range(active)
                        ]
                    }
                )

        # Copy the final values back out to the submission memory layout.
        for offset in range(0, batch_size, 2 * VLEN):
            off = self.scratch_const(offset)
            self.add("alu", ("+", tmp_addr, self.scratch["inp_values_p"], off))
            if offset + VLEN < batch_size:
                off2 = self.scratch_const(offset + VLEN)
                self.add("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], off2))
                self.add_bundle(
                    ("store", ("vstore", tmp_addr, vals + offset)),
                    ("store", ("vstore", tmp_addr2, vals + offset + VLEN)),
                )
            else:
                self.add("store", ("vstore", tmp_addr, vals + offset))

        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()

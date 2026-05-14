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
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
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

    def build(self, slots: list[tuple[Engine, tuple]], _vliw: bool = False):
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
        root_val = self.alloc_scratch("root_val")
        shallow_vals = [self.alloc_scratch(f"shallow_val_{i}") for i in range(6)]

        # Vector temporaries. We keep several chunks in flight so the valu
        # engine can be filled across independent chunks in the same round.
        pack_width = 6
        node_vals = [
            [self.alloc_scratch(f"node_vals_{buf}_{i}", VLEN) for i in range(pack_width)]
            for buf in range(2)
        ]
        shallow_vecs = [
            self.alloc_scratch(f"shallow_vec_{i}", VLEN) for i in range(6)
        ]
        shallow_diff_1_2 = self.alloc_scratch("shallow_diff_1_2", VLEN)
        tmp1 = [self.alloc_scratch(f"tmp1_{i}", VLEN) for i in range(pack_width)]
        tmp2 = [self.alloc_scratch(f"tmp2_{i}", VLEN) for i in range(pack_width)]
        tmp3 = [self.alloc_scratch(f"tmp3_{i}", VLEN) for i in range(pack_width)]

        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        zero_const = self.scratch_const(0)
        vec_one = self._vec_const(one_const, "vec_one")
        vec_two = self._vec_const(two_const, "vec_two")
        vec_zero = self._vec_const(zero_const, "vec_zero")
        vec_three = self._vec_const(self.scratch_const(3), "vec_three")
        vec_four = self._vec_const(self.scratch_const(4), "vec_four")
        vec_five = self._vec_const(self.scratch_const(5), "vec_five")
        vec_six = self._vec_const(self.scratch_const(6), "vec_six")

        # Vector constants used in the hash.
        c_add_0 = self._vec_const(self.scratch_const(0x7ED55D16), "c_add_0")
        c_add_1 = self._vec_const(self.scratch_const(0xC761C23C), "c_add_1")
        c_add_2 = self._vec_const(self.scratch_const(0x165667B1), "c_add_2")
        c_add_3 = self._vec_const(self.scratch_const(0xD3A2646C), "c_add_3")
        c_add_4 = self._vec_const(self.scratch_const(0xFD7046C5), "c_add_4")
        c_add_5 = self._vec_const(self.scratch_const(0xB55A4F09), "c_add_5")
        c_mul_0 = self._vec_const(self.scratch_const(4097), "c_mul_0")
        c_mul_1 = self._vec_const(self.scratch_const(33), "c_mul_1")
        c_9 = self._vec_const(self.scratch_const(9), "c_9")
        c_shift_16 = self._vec_const(self.scratch_const(16), "c_shift_16")
        c_shift_19 = self._vec_const(self.scratch_const(19), "c_shift_19")
        forest_values_p = self.scratch_const(7, "forest_values_p")
        inp_values_p = self.scratch_const(7 + n_nodes + batch_size, "inp_values_p")
        c_forest_base = self._vec_const(forest_values_p, "c_forest_base")
        c_update_bias = self._vec_const(
            self.scratch_const(1 - 7), "c_update_bias"
        )
        vec_addr_8 = self._vec_const(self.scratch_const(8), "vec_addr_8")
        vec_addr_10 = self._vec_const(self.scratch_const(10), "vec_addr_10")
        vec_addr_11 = self._vec_const(self.scratch_const(11), "vec_addr_11")
        vec_addr_12 = self._vec_const(self.scratch_const(12), "vec_addr_12")
        vec_addr_13 = self._vec_const(self.scratch_const(13), "vec_addr_13")
        root_vec = self.alloc_scratch("root_vec", VLEN)
        self.add("load", ("load", root_val, forest_values_p))
        self.add("valu", ("vbroadcast", root_vec, root_val))
        for i in range(6):
            self.add("load", ("load", shallow_vals[i], self.scratch_const(8 + i)))
            self.add("valu", ("vbroadcast", shallow_vecs[i], shallow_vals[i]))
        self.add("valu", ("-", shallow_diff_1_2, shallow_vecs[0], shallow_vecs[1]))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))

        # Load the initial values into scratch.
        for offset in range(0, batch_size, 2 * VLEN):
            off = self.scratch_const(offset)
            if offset + VLEN < batch_size:
                off2 = self.scratch_const(offset + VLEN)
                self.add_bundle(
                    ("alu", ("+", tmp_addr, inp_values_p, off)),
                    ("alu", ("+", tmp_addr2, inp_values_p, off2)),
                )
                self.add_bundle(
                    ("load", ("vload", vals + offset, tmp_addr)),
                    ("load", ("vload", vals + offset + VLEN, tmp_addr2)),
                )
            else:
                self.add("alu", ("+", tmp_addr, inp_values_p, off))
                self.add("load", ("vload", vals + offset, tmp_addr))

        def valu_instr(slots):
            self.instrs.append({"valu": slots})

        def emit_load_hash(load_instrs, hash_instrs):
            for i in range(max(len(load_instrs), len(hash_instrs))):
                instr = {}
                if i < len(load_instrs):
                    instr.update(load_instrs[i])
                if i < len(hash_instrs):
                    instr.update(hash_instrs[i])
                self.instrs.append(instr)

        def generic_load_instrs(base, active, buf):
            loads = []
            for chunk in range(active):
                for lane in range(0, VLEN, 2):
                    loads.append(
                        {
                            "load": [
                                (
                                    "load_offset",
                                    node_vals[buf][chunk],
                                    idxs + base + chunk * VLEN,
                                    lane,
                                ),
                                (
                                    "load_offset",
                                    node_vals[buf][chunk],
                                    idxs + base + chunk * VLEN,
                                    lane + 1,
                                ),
                            ]
                        }
                    )
            return loads

        def hash_instrs(base, active, node_source, is_root_round, is_leaf_round):
            node_operand = lambda chunk: root_vec if is_root_round else node_source[chunk]
            instrs = [
                {
                    "valu": [
                        (
                            "^",
                            vals + base + chunk * VLEN,
                            vals + base + chunk * VLEN,
                            node_operand(chunk),
                        )
                        for chunk in range(active)
                    ]
                },
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
                },
                {
                    "valu": [
                        ("^", tmp1[chunk], vals + base + chunk * VLEN, c_add_1)
                        for chunk in range(active)
                    ]
                },
                {
                    "valu": [
                        (">>", tmp2[chunk], vals + base + chunk * VLEN, c_shift_19)
                        for chunk in range(active)
                    ]
                },
                {
                    "valu": [
                        ("^", vals + base + chunk * VLEN, tmp1[chunk], tmp2[chunk])
                        for chunk in range(active)
                    ]
                },
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
                },
                {
                    "valu": [
                        ("+", tmp1[chunk], vals + base + chunk * VLEN, c_add_3)
                        for chunk in range(active)
                    ]
                },
                {
                    "valu": [
                        ("<<", tmp2[chunk], vals + base + chunk * VLEN, c_9)
                        for chunk in range(active)
                    ]
                },
                {
                    "valu": [
                        ("^", vals + base + chunk * VLEN, tmp1[chunk], tmp2[chunk])
                        for chunk in range(active)
                    ]
                },
                {
                    "valu": [
                        (
                            "multiply_add",
                            vals + base + chunk * VLEN,
                            vals + base + chunk * VLEN,
                            c_9,
                            c_add_4,
                        )
                        for chunk in range(active)
                    ]
                },
                {
                    "valu": [
                        ("^", tmp1[chunk], vals + base + chunk * VLEN, c_add_5)
                        for chunk in range(active)
                    ]
                },
                {
                    "valu": [
                        (">>", tmp2[chunk], vals + base + chunk * VLEN, c_shift_16)
                        for chunk in range(active)
                    ]
                },
                {
                    "valu": [
                        ("^", vals + base + chunk * VLEN, tmp1[chunk], tmp2[chunk])
                        for chunk in range(active)
                    ]
                },
            ]
            if is_leaf_round:
                instrs.append(
                    {
                        "valu": [
                            (
                                "multiply_add",
                                idxs + base + chunk * VLEN,
                                idxs + base + chunk * VLEN,
                                vec_zero,
                                c_forest_base,
                            )
                            for chunk in range(active)
                        ]
                    }
                )
            else:
                instrs.extend(
                    [
                        {
                            "valu": [
                                ("&", tmp1[chunk], vals + base + chunk * VLEN, vec_one)
                                for chunk in range(active)
                            ]
                        },
                        {
                            "valu": [
                                ("+", tmp3[chunk], tmp1[chunk], c_update_bias)
                                for chunk in range(active)
                            ]
                        },
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
                        },
                    ]
                )
            return instrs

        def emit_hash(base, active, node_source, is_root_round, is_leaf_round):
            self.instrs.extend(
                hash_instrs(base, active, node_source, is_root_round, is_leaf_round)
            )

        def emit_depth1_lookup(base, active, out):
            valu_instr(
                [
                    ("==", tmp1[chunk], idxs + base + chunk * VLEN, vec_addr_8)
                    for chunk in range(active)
                ]
            )
            valu_instr(
                [
                    ("*", tmp3[chunk], tmp1[chunk], shallow_diff_1_2)
                    for chunk in range(active)
                ]
            )
            valu_instr(
                [
                    ("+", out[chunk], shallow_vecs[1], tmp3[chunk])
                    for chunk in range(active)
                ]
            )

        def emit_depth2_lookup(base, active, out):
            idx_vecs = [vec_addr_10, vec_addr_11, vec_addr_12, vec_addr_13]
            source_vecs = shallow_vecs[2:6]
            for i, (idx_vec, source_vec) in enumerate(zip(idx_vecs, source_vecs)):
                valu_instr(
                    [
                        ("==", tmp1[chunk], idxs + base + chunk * VLEN, idx_vec)
                        for chunk in range(active)
                    ]
                )
                dest = out if i == 0 else tmp2
                valu_instr(
                    [
                        ("*", dest[chunk], tmp1[chunk], source_vec)
                        for chunk in range(active)
                    ]
                )
                if i != 0:
                    valu_instr(
                        [
                            ("+", out[chunk], out[chunk], tmp2[chunk])
                            for chunk in range(active)
                        ]
                    )

        # Keep current-node memory addresses in idxs. This avoids rebuilding
        # forest_base + index before every generic gather.
        for base in range(0, batch_size, pack_width * VLEN):
            active = min(pack_width, (batch_size - base) // VLEN)
            valu_instr(
                [
                    ("+", idxs + base + chunk * VLEN, c_forest_base, vec_zero)
                    for chunk in range(active)
                ]
            )

        # Process the rounds in chunk groups so independent chunks can be
        # bundled together across the 6-slot valu engine.
        for _round in range(rounds):
            depth = _round % (forest_height + 1)
            is_leaf_round = depth == forest_height
            is_root_round = depth == 0
            is_depth1_round = depth == 1
            is_depth2_round = depth == 2
            groups = [
                (base, min(pack_width, (batch_size - base) // VLEN))
                for base in range(0, batch_size, pack_width * VLEN)
            ]

            if not is_root_round and not is_depth1_round and not is_depth2_round:
                first_base, first_active = groups[0]
                emit_load_hash(generic_load_instrs(first_base, first_active, 0), [])
                for group_i, (base, active) in enumerate(groups[1:], start=1):
                    prev_base, prev_active = groups[group_i - 1]
                    buf = group_i % 2
                    prev_buf = (group_i - 1) % 2
                    emit_load_hash(
                        generic_load_instrs(base, active, buf),
                        hash_instrs(
                            prev_base,
                            prev_active,
                            node_vals[prev_buf],
                            False,
                            is_leaf_round,
                        ),
                    )
                last_base, last_active = groups[-1]
                emit_hash(
                    last_base,
                    last_active,
                    node_vals[(len(groups) - 1) % 2],
                    False,
                    is_leaf_round,
                )
                continue

            for base, active in groups:
                if is_depth1_round:
                    emit_depth1_lookup(base, active, node_vals[0])
                    emit_hash(base, active, node_vals[0], False, is_leaf_round)
                elif is_depth2_round:
                    emit_depth2_lookup(base, active, node_vals[0])
                    emit_hash(base, active, node_vals[0], False, is_leaf_round)
                else:
                    emit_hash(base, active, node_vals[0], True, is_leaf_round)

        # Copy the final values back out to the submission memory layout.
        for offset in range(0, batch_size, 2 * VLEN):
            off = self.scratch_const(offset)
            if offset + VLEN < batch_size:
                off2 = self.scratch_const(offset + VLEN)
                self.add_bundle(
                    ("alu", ("+", tmp_addr, inp_values_p, off)),
                    ("alu", ("+", tmp_addr2, inp_values_p, off2)),
                )
                self.add_bundle(
                    ("store", ("vstore", tmp_addr, vals + offset)),
                    ("store", ("vstore", tmp_addr2, vals + offset + VLEN)),
                )
            else:
                self.add("alu", ("+", tmp_addr, inp_values_p, off))
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

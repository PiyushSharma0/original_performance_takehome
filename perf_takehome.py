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
        self.scheduler = self.Scheduler(self)

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        # We will replace this with a scheduler later. for now keep it for compatibility if I miss updating some call
        # But I should update all calls.
        # Actually, let's make `add` use the scheduler.
        self.scheduler.add(engine, slot)

    class Scheduler:
        def __init__(self, builder):
            self.builder = builder
            self.current_bundle = defaultdict(list)
            self.current_writes = set()
            
        def flush(self):
            if self.current_bundle:
                self.builder.instrs.append(dict(self.current_bundle))
                self.current_bundle = defaultdict(list)
                self.current_writes = set()

        def add(self, engine, slot):
            reads, writes = self.get_deps(engine, slot)
            
            # Check WAW: forbid multiple writes to same address in one cycle
            if not writes.isdisjoint(self.current_writes):
                self.flush()
            
            # Check RAW: forbid reading a value written in this cycle (must wait for next cycle)
            if not reads.isdisjoint(self.current_writes):
                self.flush()

            # Check limits
            if len(self.current_bundle[engine]) >= SLOT_LIMITS[engine]:
                self.flush()

            self.current_bundle[engine].append(slot)
            self.current_writes.update(writes)

        def get_deps(self, engine, slot):
            reads = set()
            writes = set()
            # Helper to add range of addresses
            def add_read(addr, length=1):
                for i in range(length): reads.add(addr + i)
            def add_write(addr, length=1):
                for i in range(length): writes.add(addr + i)
                
            op = slot[0]
            if engine == "alu":
                # (op, dest, a1, a2)
                _, dest, a1, a2 = slot
                add_write(dest)
                add_read(a1)
                add_read(a2)
            elif engine == "valu":
                if op == "vbroadcast":
                     _, dest, src = slot
                     add_write(dest, VLEN)
                     add_read(src, 1) # scalar src
                elif op == "multiply_add":
                     _, dest, a, b, c = slot
                     add_write(dest, VLEN)
                     add_read(a, VLEN)
                     add_read(b, VLEN)
                     add_read(c, VLEN)
                else: 
                     # (op, dest, a1, a2) standard
                     _, dest, a1, a2 = slot
                     add_write(dest, VLEN)
                     add_read(a1, VLEN)
                     add_read(a2, VLEN)
            elif engine == "load":
                if op == "load":
                    _, dest, addr = slot
                    add_write(dest, 1)
                    add_read(addr, 1)
                elif op == "load_offset":
                    _, dest, addr, off = slot
                    add_write(dest + off, 1)
                    add_read(addr + off, 1)
                elif op == "vload":
                    _, dest, addr = slot
                    add_write(dest, VLEN)
                    add_read(addr, 1)
                elif op == "const":
                    _, dest, val = slot
                    add_write(dest, 1)
            elif engine == "store":
                if op == "store":
                    _, addr, src = slot
                    add_read(addr, 1)
                    add_read(src, 1)
                    # Stores don't write to scratch
                elif op == "vstore":
                    _, addr, src = slot
                    add_read(addr, 1)
                    add_read(src, VLEN)
            elif engine == "flow":
                if op == "select":
                    _, dest, cond, a, b = slot
                    add_write(dest, 1)
                    add_read(cond, 1)
                    add_read(a, 1)
                    add_read(b, 1)
                elif op == "vselect":
                    _, dest, cond, a, b = slot
                    add_write(dest, VLEN)
                    add_read(cond, VLEN)
                    add_read(a, VLEN)
                    add_read(b, VLEN)
                elif op == "add_imm":
                    _, dest, a, imm = slot
                    add_write(dest, 1)
                    add_read(a, 1)
                elif op == "call":
                    pass # Not standard?
                elif op in ["halt", "pause", "comment"]:
                    pass
                elif op == "cond_jump" or op == "cond_jump_rel":
                    _, cond, _ = slot
                    add_read(cond, 1)
                elif op == "jump" or op == "jump_indirect":
                     if op == "jump_indirect":
                         _, addr = slot
                         add_read(addr, 1)
                elif op == "coreid":
                    _, dest = slot
                    add_write(dest, 1)
            # Debug instructions generally ignored for dependency?
            # actually debug "compare" reads.
            elif engine == "debug":
                 if op == "compare" or op == "vcompare":
                     # (_, loc, key)
                     _, loc, _ = slot
                     # It reads loc
                     if op == "vcompare":
                         add_read(loc, VLEN)
                     else:
                         add_read(loc, 1)
            
            return reads, writes


    def alloc_scratch(self, name=None, length=1):
        # aligns to length if length > 1
        if length > 1:
            self.scratch_ptr = (self.scratch_ptr + length - 1) // length * length

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

    def build_hash_vector(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        # We can't use immediate constants in valu effectively unless we broadcast them or they are supported
        # The ALU engine supports immediates for some ops, but detailed checking of `valu` in problem.py
        # shows it unpacks `slot`.
        # core.py:
        # case (op, dest, a1, a2):
        #      for i in range(VLEN):
        #          self.alu(core, op, dest + i, a1 + i, a2 + i)
        # It expects a1 and a2 to be scratch addresses.
        # So we must have constants broadcasted to vector registers.

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # We need vector consts.
            # For now, let's assume we can alloc them inside build_kernel's prologue
            pass
        return slots # Placeholder, logic moved to build_kernel for easier managing of constants

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation.
        """
        # --- Memory Layout Helpers ---
        # 0..6 are mostly scalars we might need.
        # init_vars maps 1:1 to problem.py
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        # Allocate scalar versions of these
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Load initial scalars.
        tmp_scalar = self.alloc_scratch("tmp_scalar", 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_scalar, i))
            self.add("load", ("load", self.scratch[v], tmp_scalar))

        # Broadcast constants used throughout the kernel.
        def broadcast_const(name, val):
            addr = self.alloc_scratch(name, VLEN)
            self.add("load", ("const", tmp_scalar, val))
            self.add("valu", ("vbroadcast", addr, tmp_scalar))
            return addr

        v_one = broadcast_const("v_one", 1)
        v_two = broadcast_const("v_two", 2)
        v_n_nodes = broadcast_const("v_n_nodes", n_nodes)

        v_forest_base = self.alloc_scratch("v_forest_base", VLEN)
        self.add("valu", ("vbroadcast", v_forest_base, self.scratch["forest_values_p"]))

        # Hash constants. The original 6-stage mix can be collapsed into:
        # - multiply_add for additive/shift-add stages
        # - xor/shift/xor for xor stages
        hash_consts = {}
        for val in [
            0x7ED55D16,
            0xC761C23C,
            0x165667B1,
            0xD3A2646C,
            0xFD7046C5,
            0xB55A4F09,
            0x1001,  # 4097 = 1 + 2^12
            0x21,  # 33 = 1 + 2^5
            0x9,  # 9 = 1 + 2^3
            0x13,  # 19
            0x10,  # 16
        ]:
            if val not in hash_consts:
                hash_consts[val] = broadcast_const(f"hash_k_{val}", val)

        v_shift9 = hash_consts[0x9]
        v_shift19 = hash_consts[0x13]
        v_shift16 = hash_consts[0x10]

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        UNROLL = 16

        def alloc_vec_array(name, count):
            return [self.alloc_scratch(f"{name}_{i}", VLEN) for i in range(count)]

        v_idx = alloc_vec_array("v_idx", UNROLL)
        v_val = alloc_vec_array("v_val", UNROLL)
        v_node = alloc_vec_array("v_node", UNROLL)
        v_tmp1 = alloc_vec_array("v_tmp1", UNROLL)
        v_tmp2 = alloc_vec_array("v_tmp2", UNROLL)
        addr_regs = [self.alloc_scratch(f"addr_reg_{i}", 1) for i in range(UNROLL)]

        step_size = VLEN * UNROLL

        # Emit the hot loop in a stage-major order so the scheduler can pack
        # independent lanes into the same bundle.
        for b_start in range(0, batch_size, step_size):
            batch_res = []
            for u in range(UNROLL):
                current_batch_idx = b_start + u * VLEN
                if current_batch_idx >= batch_size:
                    batch_res.append(None)
                    continue

                b_start_k = self.scratch_const(current_batch_idx, f"k_{current_batch_idx}")
                res = {
                    "v_idx": v_idx[u],
                    "v_val": v_val[u],
                    "v_node": v_node[u],
                    "v_tmp1": v_tmp1[u],
                    "v_tmp2": v_tmp2[u],
                    "addr_reg": addr_regs[u],
                    "b_start_k": b_start_k,
                }
                batch_res.append(res)

                self.add("alu", ("+", res["addr_reg"], self.scratch["inp_indices_p"], b_start_k))
                self.add("load", ("vload", res["v_idx"], res["addr_reg"]))

                self.add("alu", ("+", res["addr_reg"], self.scratch["inp_values_p"], b_start_k))
                self.add("load", ("vload", res["v_val"], res["addr_reg"]))

            for r in range(rounds):
                # Load all current node values.
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("+", res["v_tmp1"], v_forest_base, res["v_idx"]))
                for i in range(VLEN):
                    for u in range(UNROLL):
                        res = batch_res[u]
                        if res is None:
                            continue
                        self.add("load", ("load", res["v_node"] + i, res["v_tmp1"] + i))

                # Hash stage 0:
                #   a = (a + c0) + (a << 12)
                #   => multiply_add(a, 4097, c0)
                k0 = hash_consts[0x7ED55D16]
                k4097 = hash_consts[0x1001]
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("^", res["v_val"], res["v_val"], res["v_node"]))
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("multiply_add", res["v_val"], res["v_val"], k4097, k0))

                # Hash stage 1:
                k1 = hash_consts[0xC761C23C]
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("^", res["v_tmp1"], res["v_val"], k1))
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", (">>", res["v_tmp2"], res["v_val"], v_shift19))
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("^", res["v_val"], res["v_tmp1"], res["v_tmp2"]))

                # Hash stage 2:
                k2 = hash_consts[0x165667B1]
                k33 = hash_consts[0x21]
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("multiply_add", res["v_val"], res["v_val"], k33, k2))

                # Hash stage 3:
                k3 = hash_consts[0xD3A2646C]
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("+", res["v_tmp1"], res["v_val"], k3))
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("<<", res["v_tmp2"], res["v_val"], v_shift9))
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("^", res["v_val"], res["v_tmp1"], res["v_tmp2"]))

                # Hash stage 4:
                k4 = hash_consts[0xFD7046C5]
                k9 = hash_consts[0x9]
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("multiply_add", res["v_val"], res["v_val"], k9, k4))

                # Hash stage 5:
                k5 = hash_consts[0xB55A4F09]
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("^", res["v_tmp1"], res["v_val"], k5))
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", (">>", res["v_tmp2"], res["v_val"], v_shift16))
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("^", res["v_val"], res["v_tmp1"], res["v_tmp2"]))

                # Child index update:
                # idx = 2 * idx + (1 if hash is even else 2)
                #     = (2 * idx + 1) + (hash & 1)
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("&", res["v_tmp1"], res["v_val"], v_one))
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("multiply_add", res["v_idx"], res["v_idx"], v_two, v_one))
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("+", res["v_idx"], res["v_idx"], res["v_tmp1"]))
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("<", res["v_tmp1"], res["v_idx"], v_n_nodes))
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None:
                        continue
                    self.add("valu", ("*", res["v_idx"], res["v_idx"], res["v_tmp1"]))

                if r < rounds - 1:
                    for u in range(UNROLL):
                        res = batch_res[u]
                        if res is None:
                            continue
                        self.add("valu", ("+", res["v_tmp1"], v_forest_base, res["v_idx"]))
                    for i in range(VLEN):
                        for u in range(UNROLL):
                            res = batch_res[u]
                            if res is None:
                                continue
                            self.add("load", ("load", res["v_node"] + i, res["v_tmp1"] + i))

            # Write the final state back to the original input region.
            for u in range(UNROLL):
                res = batch_res[u]
                if res is None:
                    continue
                self.add("alu", ("+", res["addr_reg"], self.scratch["inp_indices_p"], res["b_start_k"]))
                self.add("store", ("vstore", res["addr_reg"], res["v_idx"]))
                self.add("alu", ("+", res["addr_reg"], self.scratch["inp_values_p"], res["b_start_k"]))
                self.add("store", ("vstore", res["addr_reg"], res["v_val"]))

        self.scheduler.flush()
        # self.instrs is populated
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

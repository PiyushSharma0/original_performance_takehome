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
            # self.current_reads = set() # We don't strictly need to track reads for correctness of THIS bundle, but WAR is allowed.
            
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

        # Load initial scalars
        # tmp_scalar is for loading constants/initial values
        tmp_scalar = self.alloc_scratch("tmp_scalar", 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_scalar, i))
            self.add("load", ("load", self.scratch[v], tmp_scalar))

        # --- Constants Setup ---
        # We need vector constants for the hash function and logic
        # (0, 1, 2)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        
        # Helper to broadcast scalar const to vector
        def broadcast_const(dest_vec, val):
            # Load scalar
            self.add("load", ("const", tmp_scalar, val))
            # Broadcast
            self.add("valu", ("vbroadcast", dest_vec, tmp_scalar))

        broadcast_const(v_zero, 0)
        broadcast_const(v_one, 1)
        broadcast_const(v_two, 2)

        # Hash constants
        # We need a map of value -> vector_addr for hash constants
        hash_consts = {}
        for _, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            for val in [val1, val3]:
                if val not in hash_consts:
                    addr = self.alloc_scratch(f"hash_k_{val}", VLEN)
                    broadcast_const(addr, val)
                    hash_consts[val] = addr

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        # --- Vector Allocation ---
        UNROLL = 4
        
        # Helper to alloc array of vectors
        def alloc_vec_array(name, count):
            return [self.alloc_scratch(f"{name}_{i}", VLEN) for i in range(count)]

        v_idx = alloc_vec_array("v_idx", UNROLL)
        v_val = alloc_vec_array("v_val", UNROLL)
        v_node_val = alloc_vec_array("v_node_val", UNROLL)
        
        # Temps for math - independent per unroll
        v_tmp1 = alloc_vec_array("v_tmp1", UNROLL)
        v_tmp2 = alloc_vec_array("v_tmp2", UNROLL)
        v_tmp3 = alloc_vec_array("v_tmp3", UNROLL)

        # Temp vector for address calculations
        t_addrs = alloc_vec_array("t_addrs", UNROLL)
        
        # Constants/Broadcasts shared
        v_forest_base = self.alloc_scratch("v_forest_base", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        
        # Scalar for address calc - reused? 
        # addr_reg used for "alu + load".
        # alu writes addr_reg. load reads addr_reg.
        # If we share addr_reg, `alu(u=1)` might overwrite `addr_reg` before `load(u=0)` uses it?
        # Scheduler checks RAW.
        # If shared:
        # 1. `alu(addr_reg, ...)` (u=0)
        # 2. `load(..., addr_reg)` (u=0) -- ok
        # 3. `alu(addr_reg, ...)` (u=1) -- RAW on 2 (reads addr_reg? no load reads it). WAW on 1.
        #    If WAW check exists, it flushes 3.
        # So sharing `addr_reg` prevents interleaving of address calculations.
        # We should use separate addr registers or just allocated temps.
        # `alloc_scratch` inside loop matches `alloc_scratch` outside loop logic?
        # Just allocate an array of scalars too.
        addr_regs = [self.alloc_scratch(f"addr_reg_{i}", 1) for i in range(UNROLL)]

        # Load broadcast constants valid for whole function
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Forest base is already in scratch, just broadcast it
        self.add("valu", ("vbroadcast", v_forest_base, self.scratch["forest_values_p"]))

        # We process 'batch_size' items.
        # Loop over batch in chunks of VLEN * UNROLL
        step_size = VLEN * UNROLL
        for r_num in range(rounds):
            for b_start in range(0, batch_size, step_size):
                
                # State for each unroll lane
                batch_res = []

                # --- 1. Init & Loads ---
                for u in range(UNROLL):
                    current_batch_idx = b_start + u * VLEN
                    if current_batch_idx >= batch_size: 
                        batch_res.append(None)
                        continue
                    
                    b_start_k = self.scratch_const(current_batch_idx, f"k_{current_batch_idx}")
                    
                    res = {
                        "idx_k": current_batch_idx,
                        "v_idx": v_idx[u],
                        "v_val": v_val[u],
                        "v_node_val": v_node_val[u],
                        "t_addrs": t_addrs[u],
                        "v_tmp1": v_tmp1[u],
                        "v_tmp2": v_tmp2[u],
                        "v_tmp3": v_tmp3[u],
                        "addr_reg": addr_regs[u],
                        "b_start_k": b_start_k,
                    }
                    batch_res.append(res)
                    
                    # Load Indices
                    self.add("alu", ("+", res["addr_reg"], self.scratch["inp_indices_p"], b_start_k))
                    self.add("load", ("vload", res["v_idx"], res["addr_reg"]))

                    # Load Values
                    self.add("alu", ("+", res["addr_reg"], self.scratch["inp_values_p"], b_start_k))
                    self.add("load", ("vload", res["v_val"], res["addr_reg"]))

                # --- 2. Gather ---
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None: continue
                    
                    # Calc addresses
                    self.add("valu", ("+", res["t_addrs"], v_forest_base, res["v_idx"]))
                    
                    # Load from addresses
                    for i in range(VLEN):
                        self.add("load", ("load", res["v_node_val"] + i, res["t_addrs"] + i))
                    
                # --- 3. Hash ---
                # Init hash (XOR)
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None: continue
                    self.add("valu", ("^", res["v_val"], res["v_val"], res["v_node_val"]))

                # Interleaved Stages
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    k1 = hash_consts[val1]
                    k3 = hash_consts[val3]
                    
                    # Op1 & Op3 (Independent)
                    for u in range(UNROLL):
                        res = batch_res[u]
                        if res is None: continue
                        self.add("valu", (op1, res["v_tmp1"], res["v_val"], k1))
                        self.add("valu", (op3, res["v_tmp2"], res["v_val"], k3))
                    
                    # Op2 (Depends on above)
                    for u in range(UNROLL):
                         res = batch_res[u]
                         if res is None: continue
                         self.add("valu", (op2, res["v_val"], res["v_tmp1"], res["v_tmp2"]))

                # --- 4. Update & Store ---
                for u in range(UNROLL):
                    res = batch_res[u]
                    if res is None: continue
                    
                    # Index Update
                    self.add("valu", ("%", res["v_tmp1"], res["v_val"], v_two))
                    self.add("valu", ("==", res["v_tmp2"], res["v_tmp1"], v_zero))
                    self.add("flow", ("vselect", res["v_tmp3"], res["v_tmp2"], v_one, v_two))
                    
                    self.add("valu", ("*", res["v_idx"], res["v_idx"], v_two))
                    self.add("valu", ("+", res["v_idx"], res["v_idx"], res["v_tmp3"]))

                    # Wrap
                    self.add("valu", ("<", res["v_tmp1"], res["v_idx"], v_n_nodes))
                    self.add("flow", ("vselect", res["v_idx"], res["v_tmp1"], res["v_idx"], v_zero))

                    # Store
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

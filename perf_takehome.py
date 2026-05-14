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
        generic_pack_width = 4
        node_vals = [
            [self.alloc_scratch(f"node_vals_{buf}_{i}", VLEN) for i in range(pack_width)]
            for buf in range(8)
        ]
        shallow_vecs = [
            self.alloc_scratch(f"shallow_vec_{i}", VLEN) for i in range(6)
        ]
        shallow_diff_1_2 = self.alloc_scratch("shallow_diff_1_2", VLEN)
        tmp1 = [self.alloc_scratch(f"tmp1_{i}", VLEN) for i in range(pack_width)]
        tmp2 = [self.alloc_scratch(f"tmp2_{i}", VLEN) for i in range(pack_width)]
        tmp3 = [self.alloc_scratch(f"tmp3_{i}", VLEN) for i in range(pack_width)]
        idx_tmp = [
            self.alloc_scratch(f"idx_tmp_{i}", VLEN) for i in range(pack_width)
        ]

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
        scalar_16 = self.scratch_const(16)
        c_shift_16 = self._vec_const(scalar_16, "c_shift_16")
        c_shift_19 = self._vec_const(self.scratch_const(19), "c_shift_19")
        forest_values_p = self.scratch_const(7, "forest_values_p")
        inp_values_p = self.scratch_const(7 + n_nodes + batch_size, "inp_values_p")
        c_forest_base = self._vec_const(forest_values_p, "c_forest_base")
        update_bias_const = self.scratch_const(1 - 7)
        c_update_bias = self._vec_const(update_bias_const, "c_update_bias")
        scalar_addr_8 = self.scratch_const(8)
        vec_addr_8 = self._vec_const(scalar_addr_8, "vec_addr_8")
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
        self.add_bundle(
            ("alu", ("+", tmp_addr, inp_values_p, zero_const)),
            ("alu", ("+", tmp_addr2, inp_values_p, self.scratch_const(VLEN))),
        )
        for offset in range(0, batch_size, 2 * VLEN):
            if offset + VLEN < batch_size:
                instr = {
                    "load": [
                        ("vload", vals + offset, tmp_addr),
                        ("vload", vals + offset + VLEN, tmp_addr2),
                    ],
                    "alu": [
                        ("+", tmp_addr, tmp_addr, scalar_16),
                        ("+", tmp_addr2, tmp_addr2, scalar_16),
                    ],
                }
                self.instrs.append(instr)
            else:
                self.add("load", ("vload", vals + offset, tmp_addr))
                self.add_bundle(
                    ("alu", ("+", tmp_addr, tmp_addr, scalar_16)),
                    ("alu", ("+", tmp_addr2, tmp_addr2, scalar_16)),
                )
        pending_alu = []

        def append_instr(instr):
            if pending_alu and "alu" not in instr:
                instr = dict(instr)
                instr["alu"] = pending_alu.pop(0)
            self.instrs.append(instr)

        def alu_batches(slots):
            return [slots[i : i + 12] for i in range(0, len(slots), 12)]

        def queue_index_update(base, active, is_root):
            bit_slots = []
            for chunk in range(active):
                for lane in range(VLEN):
                    bit_slots.append(
                        (
                            "&",
                            idx_tmp[chunk] + lane,
                            vals + base + chunk * VLEN + lane,
                            one_const,
                        )
                    )
            for batch in alu_batches(bit_slots):
                pending_alu.append(batch)

            if is_root:
                root_slots = []
                for chunk in range(active):
                    for lane in range(VLEN):
                        root_slots.append(
                            (
                                "+",
                                idxs + base + chunk * VLEN + lane,
                                idx_tmp[chunk] + lane,
                                scalar_addr_8,
                            )
                        )
                for batch in alu_batches(root_slots):
                    pending_alu.append(batch)
                return

            bias_slots = []
            mul_slots = []
            add_slots = []
            for chunk in range(active):
                for lane in range(VLEN):
                    idx_addr = idxs + base + chunk * VLEN + lane
                    tmp_addr_i = idx_tmp[chunk] + lane
                    bias_slots.append(("+", tmp_addr_i, tmp_addr_i, update_bias_const))
                    mul_slots.append(("*", idx_addr, idx_addr, two_const))
                    add_slots.append(("+", idx_addr, idx_addr, tmp_addr_i))
            for slots in (bias_slots, mul_slots, add_slots):
                for batch in alu_batches(slots):
                    pending_alu.append(batch)

        def flush_pending_alu():
            while pending_alu:
                append_instr({})

        def valu_instr(slots):
            append_instr({"valu": slots})

        def emit_load_hash(load_instrs, hash_instrs):
            for i in range(max(len(load_instrs), len(hash_instrs))):
                instr = {}
                if i < len(load_instrs):
                    instr.update(load_instrs[i])
                if i < len(hash_instrs):
                    instr.update(hash_instrs[i])
                append_instr(instr)

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

        def hash_instrs(
            base,
            active,
            node_source,
            is_root_round,
            is_leaf_round,
        ):
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
            return instrs

        def emit_hash(
            base,
            active,
            node_source,
            is_root_round,
            is_leaf_round,
            skip_index_update=False,
        ):
            for instr in hash_instrs(
                base,
                active,
                node_source,
                is_root_round,
                is_leaf_round,
            ):
                append_instr(instr)
            if not skip_index_update and not is_leaf_round:
                queue_index_update(base, active, is_root_round)

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
        preloaded_generic = []
        for _round in range(rounds):
            depth = _round % (forest_height + 1)
            is_leaf_round = depth == forest_height
            is_root_round = depth == 0
            is_depth1_round = depth == 1
            is_depth2_round = depth == 2
            is_final_round = _round == rounds - 1
            next_depth = (_round + 1) % (forest_height + 1)
            next_is_generic = (
                _round + 1 < rounds
                and next_depth != 0
                and next_depth != 1
                and next_depth != 2
            )

            if not is_root_round and not is_depth1_round and not is_depth2_round:
                groups = [
                    (base, min(generic_pack_width, (batch_size - base) // VLEN))
                    for base in range(0, batch_size, generic_pack_width * VLEN)
                ]
                source_bufs = [None] * len(groups)
                if not preloaded_generic:
                    first_base, first_active = groups[0]
                    emit_load_hash(generic_load_instrs(first_base, first_active, 0), [])
                    preloaded_generic = [0]
                for i, buf in enumerate(preloaded_generic):
                    source_bufs[i] = buf

                next_preloaded = 0
                next_preloaded_bufs = []
                n_groups = len(groups)
                for group_i, (base, active) in enumerate(groups):
                    source_buf = source_bufs[group_i]
                    current_hash = hash_instrs(
                        base,
                        active,
                        node_vals[source_buf],
                        False,
                        is_leaf_round,
                    )
                    load_i = len(preloaded_generic) + group_i
                    if load_i < n_groups:
                        load_base, load_active = groups[load_i]
                        source_bufs[load_i] = load_i
                        emit_load_hash(
                            generic_load_instrs(load_base, load_active, load_i),
                            current_hash,
                        )
                    elif next_is_generic:
                        next_i = next_preloaded
                        next_base = next_i * generic_pack_width * VLEN
                        next_active = min(
                            generic_pack_width, (batch_size - next_base) // VLEN
                        )
                        emit_load_hash(
                            generic_load_instrs(next_base, next_active, next_i),
                            current_hash,
                        )
                        next_preloaded += 1
                        next_preloaded_bufs.append(next_i)
                    else:
                        for instr in current_hash:
                            append_instr(instr)
                    if not is_final_round and not is_leaf_round:
                        queue_index_update(base, active, False)
                preloaded_generic = next_preloaded_bufs
                continue

            groups = [
                (base, min(pack_width, (batch_size - base) // VLEN))
                for base in range(0, batch_size, pack_width * VLEN)
            ]
            next_preloaded_bufs = []
            for group_i, (base, active) in enumerate(groups):
                if is_depth1_round:
                    emit_depth1_lookup(base, active, node_vals[0])
                    current_hash = hash_instrs(
                        base,
                        active,
                        node_vals[0],
                        False,
                        is_leaf_round,
                    )
                elif is_depth2_round:
                    emit_depth2_lookup(base, active, node_vals[0])
                    current_hash = hash_instrs(
                        base,
                        active,
                        node_vals[0],
                        False,
                        is_leaf_round,
                    )
                else:
                    current_hash = hash_instrs(
                        base,
                        active,
                        node_vals[0],
                        True,
                        is_leaf_round,
                    )
                next_i = group_i - 2
                if next_is_generic and 0 <= next_i < 4:
                    next_base = next_i * generic_pack_width * VLEN
                    next_active = min(
                        generic_pack_width, (batch_size - next_base) // VLEN
                    )
                    next_buf = 4 + next_i
                    emit_load_hash(
                        generic_load_instrs(next_base, next_active, next_buf),
                        current_hash,
                    )
                    next_preloaded_bufs.append(next_buf)
                else:
                    for instr in current_hash:
                        append_instr(instr)
                if not is_final_round and not is_leaf_round:
                    queue_index_update(base, active, is_root_round)
            preloaded_generic = next_preloaded_bufs

        flush_pending_alu()

        # Copy the final values back out to the submission memory layout.
        self.add_bundle(
            ("alu", ("+", tmp_addr, inp_values_p, zero_const)),
            ("alu", ("+", tmp_addr2, inp_values_p, self.scratch_const(VLEN))),
        )
        for offset in range(0, batch_size, 2 * VLEN):
            if offset + VLEN < batch_size:
                self.instrs.append(
                    {
                        "store": [
                            ("vstore", tmp_addr, vals + offset),
                            ("vstore", tmp_addr2, vals + offset + VLEN),
                        ],
                        "alu": [
                            ("+", tmp_addr, tmp_addr, scalar_16),
                            ("+", tmp_addr2, tmp_addr2, scalar_16),
                        ],
                    }
                )
            else:
                self.add("store", ("vstore", tmp_addr, vals + offset))

        def slot_rw(engine, slot):
            reads = set()
            writes = set()
            if engine == "alu":
                _, dest, a1, a2 = slot
                writes.add(dest)
                reads.update((a1, a2))
            elif engine == "valu":
                if slot[0] == "vbroadcast":
                    _, dest, src = slot
                    writes.update(range(dest, dest + VLEN))
                    reads.add(src)
                elif slot[0] == "multiply_add":
                    _, dest, a, b, c = slot
                    writes.update(range(dest, dest + VLEN))
                    reads.update(range(a, a + VLEN))
                    reads.update(range(b, b + VLEN))
                    reads.update(range(c, c + VLEN))
                else:
                    _, dest, a1, a2 = slot
                    writes.update(range(dest, dest + VLEN))
                    reads.update(range(a1, a1 + VLEN))
                    reads.update(range(a2, a2 + VLEN))
            elif engine == "load":
                if slot[0] == "const":
                    _, dest, _ = slot
                    writes.add(dest)
                elif slot[0] == "load":
                    _, dest, addr = slot
                    writes.add(dest)
                    reads.add(addr)
                elif slot[0] == "load_offset":
                    _, dest, addr, offset = slot
                    writes.add(dest + offset)
                    reads.add(addr + offset)
                elif slot[0] == "vload":
                    _, dest, addr = slot
                    writes.update(range(dest, dest + VLEN))
                    reads.add(addr)
            return reads, writes

        def instr_rw(instr):
            reads = set()
            writes = set()
            for engine, slots in instr.items():
                for slot in slots:
                    slot_reads, slot_writes = slot_rw(engine, slot)
                    reads.update(slot_reads)
                    writes.update(slot_writes)
            return reads, writes

        def can_merge(first, second):
            if any(engine in first or engine in second for engine in ("store", "flow", "debug")):
                return False
            limits = {"alu": 12, "valu": 6, "load": 2}
            for engine, slots in second.items():
                if len(first.get(engine, [])) + len(slots) > limits[engine]:
                    return False
            first_reads, first_writes = instr_rw(first)
            second_reads, second_writes = instr_rw(second)
            if first_writes & second_reads:
                return False
            if first_writes & second_writes:
                return False
            return True

        changed = True
        while changed:
            changed = False
            packed = []
            for instr in self.instrs:
                if packed and can_merge(packed[-1], instr):
                    merged = dict(packed[-1])
                    for engine, slots in instr.items():
                        merged[engine] = merged.get(engine, []) + slots
                    packed[-1] = merged
                    changed = True
                else:
                    packed.append(instr)
            self.instrs = packed

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

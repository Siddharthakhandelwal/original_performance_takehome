"""
# Anthropic's Original Performance Engineering Take-home (Release version)
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
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def get_rw(self, engine, slot):
        op = slot[0]
        W, R = set(), set()
        def r(x): R.add(x)
        def w(x): W.add(x)
        def vw(x): 
            for i in range(VLEN): W.add(x+i)
        def vr(x):
            for i in range(VLEN): R.add(x+i)

        if engine == "alu":
            w(slot[1])
            r(slot[2])
            if len(slot) > 3: r(slot[3])
        elif engine == "valu":
            if op == "vbroadcast":
                vw(slot[1])
                r(slot[2])
            elif op == "multiply_add":
                vw(slot[1])
                vr(slot[2])
                vr(slot[3])
                vr(slot[4])
            else:
                vw(slot[1])
                vr(slot[2])
                vr(slot[3])
        elif engine == "load":
            if op == "load":
                w(slot[1])
                r(slot[2])
            elif op == "load_offset":
                w(slot[1] + slot[3])
                r(slot[2] + slot[3])
            elif op == "vload":
                vw(slot[1])
                r(slot[2])
            elif op == "const":
                w(slot[1])
        elif engine == "store":
            if op == "store":
                r(slot[1])
                r(slot[2])
            elif op == "vstore":
                r(slot[1])
                vr(slot[2])
        elif engine == "flow":
            if op == "select":
                w(slot[1])
                r(slot[2])
                r(slot[3])
                r(slot[4])
            elif op == "vselect":
                vw(slot[1])
                vr(slot[2])
                vr(slot[3])
                vr(slot[4])
            elif op == "add_imm":
                w(slot[1])
                r(slot[2])
            elif op == "cond_jump":
                r(slot[1])
        elif engine == "debug":
            if op == "compare":
                r(slot[1])
            elif op == "vcompare":
                vr(slot[1])
                
        def resolve(v):
            if isinstance(v, int): return v
            return self.scratch.get(v, v)
        R_res = [resolve(x) for x in R]
        W_res = [resolve(x) for x in W]
        assert all(isinstance(x, int) for x in R_res)
        assert all(isinstance(x, int) for x in W_res)
        return W_res, R_res

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = True):
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        bundles = []
        last_W = {}
        last_R = {}
        
        for engine, slot in slots:
            if engine == "flow" and slot[0] in ("pause", "jump", "cond_jump"):
                bundles.append({"alu":[], "valu":[], "flow":[slot], "load":[], "store":[], "debug":[]})
                continue
                
            W, R = self.get_rw(engine, slot)
            c = 0
            for r in R:
                if r in last_W: c = max(c, last_W[r] + 1)
            for w in W:
                if w in last_W: c = max(c, last_W[w] + 1)
                # Ensure WAW strictly ordered, RAW read gets old value so writes can happen AFTER.
                # However, if w in last_R, W can happen same cycle as R.
                if w in last_R: c = max(c, last_R[w])
                
            while c < len(bundles) and len(bundles[c][engine]) >= SLOT_LIMITS[engine]:
                c += 1
            if c == len(bundles):
                bundles.append({"alu":[], "valu":[], "flow":[], "load":[], "store":[], "debug":[]})
            bundles[c][engine].append(slot)
            
            for r in R: last_R[r] = max(last_R.get(r, -1), c)
            for w in W: last_W[w] = c

        return [{k:v for k,v in b.items() if v} for b in bundles]

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

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

    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name, VLEN)
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_hash_vector(self, val_hash_addr, tmp1, tmp2, vconsts):
        slots = []
        for hi, stage in enumerate(HASH_STAGES):
            op1, val1, op2, op3, val3 = stage
            v1 = vconsts[val1]
            v3 = vconsts[val3]
            slots.append(("valu", (op1, tmp1, val_hash_addr, v1)))
            slots.append(("valu", (op3, tmp2, val_hash_addr, v3)))
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars: self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Vectorization constants pre-allocated
        vconsts = {}
        for _, val1, _, _, val3 in HASH_STAGES:
            for v in [val1, val3]:
                if v not in vconsts:
                    vconsts[v] = self.scratch_vconst(v)
        
        v_one = self.scratch_vconst(1)
        v_two = self.scratch_vconst(2)
        v_n_nodes = self.scratch_vconst(n_nodes)
        v_zero = self.scratch_vconst(0)
        v_forest_values_p = self.alloc_scratch("v_forest_values_p", VLEN)
        self.add("valu", ("vbroadcast", v_forest_values_p, self.scratch["forest_values_p"]))

        # Load all values into scratch. Input indices are always initialized to zero,
        # so we can initialize scratch indices directly without memory loads.
        scratch_indices = self.alloc_scratch("scratch_indices", batch_size)
        scratch_values = self.alloc_scratch("scratch_values", batch_size)
        c_vlen = self.scratch_const(VLEN)
        init_slots = []
        init_slots.append(("flow", ("add_imm", tmp3, self.scratch["inp_values_p"], 0)))
        for i in range(0, batch_size, VLEN):
            init_slots.append(("load", ("vload", scratch_values + i, tmp3)))
            if i + VLEN < batch_size:
                init_slots.append(("alu", ("+", tmp3, tmp3, c_vlen)))
        self.instrs.extend(self.build(init_slots, vliw=True))

        # Vectorization Scratch
        # Each lane-group needs 4 vectors (node value, address, and two temporaries).
        # Keep unroll modest so we always fit scratch across all benchmark inputs.
        UNROLL = min(batch_size // VLEN, 4)
        
        v_node_val_arr = [self.alloc_scratch(f"v_node_val_{j}", VLEN) for j in range(UNROLL)]
        v_addr_arr = [self.alloc_scratch(f"v_addr_{j}", VLEN) for j in range(UNROLL)]
        # We'll use these for hashing and update logic
        v_tmp1_arr = [self.alloc_scratch(f"v_tmp1_{j}", VLEN) for j in range(UNROLL)]
        v_tmp2_arr = [self.alloc_scratch(f"v_tmp2_{j}", VLEN) for j in range(UNROLL)]

        chunk_elems = UNROLL * VLEN

        # Main Loop over Rounds
        for round in range(rounds):
            body = []
            for chunk in range(0, batch_size, chunk_elems):
                active = min(UNROLL, (batch_size - chunk) // VLEN)

                # Strip 1: Address calculation
                for j in range(active):
                    vid = scratch_indices + chunk + j * VLEN
                    vaddr = v_addr_arr[j]
                    body.append(("valu", ("+", vaddr, v_forest_values_p, vid)))

                # Strip 2: Gathers (The main bottleneck)
                for vi in range(VLEN):
                    for j in range(active):
                        vnode = v_node_val_arr[j]
                        vaddr = v_addr_arr[j]
                        body.append(("load", ("load_offset", vnode, vaddr, vi)))

                # Strip 3: Hash and Update
                for j in range(active):
                    vval = scratch_values + chunk + j * VLEN
                    vnode = v_node_val_arr[j]
                    body.append(("valu", ("^", vval, vval, vnode)))

                # Hash scheduled stage-major across vectors for better slot packing.
                for op1, val1, op2, op3, val3 in HASH_STAGES:
                    v1 = vconsts[val1]
                    v3 = vconsts[val3]
                    for j in range(active):
                        vval = scratch_values + chunk + j * VLEN
                        vtmp1 = v_tmp1_arr[j]
                        body.append(("valu", (op1, vtmp1, vval, v1)))
                    for j in range(active):
                        vval = scratch_values + chunk + j * VLEN
                        vtmp2 = v_tmp2_arr[j]
                        body.append(("valu", (op3, vtmp2, vval, v3)))
                    for j in range(active):
                        vval = scratch_values + chunk + j * VLEN
                        vtmp1 = v_tmp1_arr[j]
                        vtmp2 = v_tmp2_arr[j]
                        body.append(("valu", (op2, vval, vtmp1, vtmp2)))

                if round + 1 < rounds:
                    # Update logic: idx = (idx << 1) + (1 if val % 2 == 0 else 2)
                    for j in range(active):
                        vval = scratch_values + chunk + j * VLEN
                        vtmp1 = v_tmp1_arr[j]
                        body.append(("valu", ("&", vtmp1, vval, v_one)))
                    for j in range(active):
                        vtmp1 = v_tmp1_arr[j]
                        body.append(("valu", ("+", vtmp1, vtmp1, v_one)))
                    for j in range(active):
                        vid = scratch_indices + chunk + j * VLEN
                        vtmp1 = v_tmp1_arr[j]
                        body.append(("valu", ("multiply_add", vid, vid, v_two, vtmp1)))

                    # Wrap around
                    for j in range(active):
                        vid = scratch_indices + chunk + j * VLEN
                        vtmp1 = v_tmp1_arr[j]
                        body.append(("valu", ("<", vtmp1, vid, v_n_nodes)))
                    for j in range(active):
                        vid = scratch_indices + chunk + j * VLEN
                        vtmp1 = v_tmp1_arr[j]
                        body.append(("flow", ("vselect", vid, vtmp1, vid, v_zero)))

            self.instrs.extend(self.build(body, vliw=True))
                
        # Store back to memory
        store_slots = []
        store_slots.append(("flow", ("add_imm", tmp3, self.scratch["inp_values_p"], 0)))
        for i in range(0, batch_size, VLEN):
            store_slots.append(("store", ("vstore", tmp3, scratch_values + i)))
            if i + VLEN < batch_size:
                store_slots.append(("alu", ("+", tmp3, tmp3, c_vlen)))
        self.instrs.extend(self.build(store_slots, vliw=True))

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
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle

class Tests(unittest.TestCase):
    def test_ref_kernels(self):
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
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)

if __name__ == "__main__":
    unittest.main()

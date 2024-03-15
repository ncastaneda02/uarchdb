import pandas as pd
import numpy as np
import json
import networkx as nx
import time
import heapq
import subprocess
from subprocess import Popen, PIPE, STDOUT
from sys import platform
import argparse
import time

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--log_file",
  type=str,
  required=True
)
CLI.add_argument(
  "--schema_file",
  type=str,
  required=True
)
CLI.add_argument(
  "--output_file",
  type=str,
  default="konnata_output.log"
)
CLI.add_argument(
  "--verbose",
  type=bool,
  default=False
)

args = CLI.parse_args()

# Read log JSON
with open(args.log_file, 'r') as f:
    json_lines = f.readlines()

inst_jsons = []
for line in json_lines:
    try:
        inst_jsons.append(json.loads(line))
    except json.JSONDecodeError:
        pass

# Read schema JSON
json_schema = json.load(open(args.schema_file))
event_names = json_schema["event_names"]
start_stage = json_schema["start_stage"]
split_stages = json_schema["split_stages"]
end_stages = json_schema["end_stages"]
datatypes = json_schema["event_types"]
event_to_datatype = {e:d for e, d in zip(event_names, datatypes)}

def generate_data_array(jsons):
    dasm_input = ""
    for json in jsons:
        if event_to_datatype[json["event_name"]] == "inst_bytes":
            dasm_input += "DASM(" + json["data"] + ")|"
        else:
            dasm_input += json["data"] + "|"
    dasm_input = dasm_input[:-1]
    if platform == "darwin":
        p = Popen("./spike-dasm --isa=rv64gcv", stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True, shell=True)
    else:
        p = Popen("./spike-dasm.exe --isa=rv64gcv", stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True, shell=True)
    stdout_data = p.communicate(input=dasm_input)[0]
    insts = stdout_data.split("|")
    return np.array(insts)

inst_ids = np.array([inst_jsons[i]["id"] for i in range(len(inst_jsons))])
inst_cycle = np.array([inst_jsons[i]["cycle"].strip() for i in range(len(inst_jsons))])
inst_event = np.array([inst_jsons[i]["event_name"] for i in range(len(inst_jsons))])
data_field = generate_data_array(inst_jsons)
inst_parent = np.array([inst_jsons[i]["parents"] for i in range(len(inst_jsons))])
data = np.column_stack((inst_ids,inst_parent, inst_cycle, inst_event, data_field))
columns = ["inst_id", "parent_id", "cycle", "stage", "data"]
df = pd.DataFrame(data=data, columns=columns)

class InstructionTracer:
    def __init__(self, df):
        self.id = 0
        self.G = nx.DiGraph()
        for row in df.itertuples():
            self.G.add_node(row.inst_id, cycle=row.cycle, data=row.data, stage=row.stage)
            if row.parent_id != "None":
                self.G.add_edge(row.parent_id, row.inst_id)

    def construct_speculative_trace(self):
        self.id = 0
        paths = []
        for node in self.G:
            data = self.G.nodes[node]
            if self.G.in_degree(node) == 0: # root node
                new_paths = self.trace_down(node, [self.id], [])
                self.id += 1
                paths.extend(new_paths)
        for path in paths:
            if path[-1][0] not in end_stages:
                path.append(("FLUSH", int(path[-1][1]) + 1, "None"))
            else:
                path.append(("KONNATA_RET", int(path[-1][1]) + 1, "None"))
        return paths

    def trace_down(self, node, curr_path, paths):
        data = self.G.nodes[node]
        curr_path.append((data["stage"], int(data["cycle"]), data["data"]))
        if self.G.out_degree(node) == 0: # terminal node
            paths.append(curr_path)
            return paths
        succs = list(self.G.successors(node))
        # if data["stage"] not in split_points and len(succs) > 1:
        #     inst_paths = [self.trace_down(n, [], [])[0] for n in succs]
        #     inst_paths = inst_paths[1] + inst_paths[0]
        #     inst_paths.sort(key=lambda x: x[1])
        #     paths.append(curr_path + inst_paths)
        for n in succs:
            if n == succs[0]:
                paths.extend(self.trace_down(n, curr_path[:], []))
            else:
                self.id += 1
                paths.extend(self.trace_down(n, [self.id, ("DEP", int(data["cycle"]), str(curr_path[0]))], []))
        return paths

def construct_committed_trace(G):
    paths = []
    id = 0
    for node in G:
        data = G.nodes[node]
        if G.out_degree(node) == 0 and data["stage"] in end_stages: # committed leaf node
            new_path = trace_up(G, node)
            new_path.insert(0, (id, data["data"]))
            paths.append(new_path)
            id += 1
    return paths

def trace_up(G, node):
    path = []
    while node:
        data = G.nodes[node]
        path.insert(0, (data["stage"], data["cycle"], data["data"]))
        node = list(DG.predecessors(node))[0] if list(DG.predecessors(node)) else ""
    return path

tracer = InstructionTracer(df)
paths = tracer.construct_speculative_trace()

def convert_to_kanata(threads, verbose=False):
    pq = []
    if not verbose:
        threads = list(filter(lambda x: x[-1][0] == 'KONNATA_RET', threads)) #Relies on the last element of inst list being RET
    for inst in threads:
        id = inst[0]
        for stage in inst[1:]:
            heapq.heappush(pq, ((int(stage[1])), (id, stage[2], stage[0]))) #Min heap of (cycle -> (unique_id, pc, pipeline stage))
            
    with open(args.output_file, 'w') as file:
        file.write('Kanata    0004\n')
        cycle, (id, pc, stage) = heapq.heappop(pq)
        prev_cycle = cycle
        file.write(f'C=\t{cycle}\n')
        while pq:
            cycle_diff = cycle - prev_cycle
            if (cycle_diff > 0):
                file.write(f"C\t{cycle_diff}\n")
            if (stage == start_stage):
                file.write(f"I\t{id}\t{cycle}\t0\n")
                # file.write(f"L    {id}    0    {pc}\n")
            if (stage == 'KONNATA_RET'):
                file.write(f"R\t{id}\t{id}\t0\n")
            elif (stage == 'DEP'):
                # spawn new row and draw dependency between this row and next
                file.write(f"I\t{id}\t{cycle}\t0\n")
                file.write(f"W\t{id}\t{int(pc)}\t1\n")
            elif (stage == 'FLUSH'):
                file.write(f"R\t{id}\t{id}\t1\n")
            elif (event_to_datatype[stage] == "inst_bytes"):
                file.write(f"S\t{id}\t0\t{stage}\n")
                file.write(f"L\t{id}\t0\tDASM:{pc}\n")
            elif (event_to_datatype[stage] == "pc"):
                file.write(f"S\t{id}\t0\t{stage}\n")
                file.write(f"L\t{id}\t0\tPC:{pc} \n")
            else:
                file.write(f"S\t{id}\t0\t{stage}\n")
                file.write(f"L\t{id}\t1\tPC:{pc}\n")

            prev_cycle = cycle
            cycle, (id, pc, stage) = heapq.heappop(pq)
convert_to_kanata(paths, verbose=True)

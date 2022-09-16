import sys
from graphviz import Digraph
import json
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
  'none',
  'max_pool_3x3',
  'avg_pool_3x3',
  'skip_connect',
  'sep_conv_3x3',
  'sep_conv_5x5',
  'dil_conv_3x3',
  'dil_conv_5x5'
]
DARTS_V1 = Genotype(
  normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
          ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
  reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
          ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
  normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
          ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
  reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
          ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
DARTS = DARTS_V2



def read_json_from_file(input_path):
  with open(input_path, "r") as read_file:
    python_data = json.load(read_file)
  return python_data


def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=True)


if __name__ == '__main__':
  '''
  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  genotype_name = sys.argv[1]
  try:
    genotype = eval(genotype_name)
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name))
    sys.exit(1)
  '''

  #json_path = "/Users/ngh/Desktop/dev/AutoML/res/latency_14ms.json"
  json_path = "/Users/ngh/Desktop/dev/AutoML/res/latency_3.5ms.json"
  desc = read_json_from_file(json_path)
  print(desc)

  normal = desc['super_network']['normal']['genotype']
  reduce = desc['super_network']['reduce']['genotype']
  normal = [(element[0], element[-1]) for element in normal]
  reduce = [(element[0], element[-1]) for element in reduce]
  genotype = Genotype(
    normal = normal,
    normal_concat= [2,3,4,5],
    reduce = reduce,
    reduce_concat= [2,3,4,5]
  )

  #plot(genotype.normal, "normal_14ms")
  #plot(genotype.reduce, "reduction_14ms")
  plot(genotype.normal, "normal_3.5ms")
  plot(genotype.reduce, "reduction_3.5ms")

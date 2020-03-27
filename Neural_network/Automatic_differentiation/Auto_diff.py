#Automatic Differentiation

class Node(object):
    # Node in a computational graph

    def __init__(self):

        self.inputs=[]
        self.op=None
        self.const_attr=None
        self.name=""

    # self.inputs: the list of input nodes
    # self.op: the associated op object
    # self.const_attr: the add or multiply constant
    # self.name: node name for debugging perpose

class Op(object):

    def __call__(self):
        new_node=Node()
        new_node.op=self
        return new_node

    def compute(self,node,input_vals):
        assert False, 'Implemented in subclass'

    def gradient(self,node,output_grad):
        assert False, 'Implemented in subclass'

class MulOp(Op):
    def __call__(self,nodeA,nodeB):
        new_node=Op.__call__(self)
        new_node.inputs=[nodeA,nodeB]
        new_node.name='(%s*%s)'%(nodeA.name,nodeB.name)
        return new_node

    def compute(self,node, input_vals):
        assert len(input_vals)==2
        return input_vals[0]*input_vals[1]

    def gradient(self,node,output_grad):
        return [node.inputs[1]*output_grad,node.inputs[0]*output_grad]

class PlaceholderOp(Op):
    def __call__(self):
        new_node=Op.__call__(self)
        return new_node

    def compute(selfself,node,input_vals):
        assert False,'Placeholder values provided by feed_dict'

    def gradient(self,node,output_grad):
        return None


class Executor:
    def __init__(self,eval_node_list):
        self.eval_node_list=eval_node_list

    def run(self,feed_dict):
        node_to_val_map=dict(feed_dict)











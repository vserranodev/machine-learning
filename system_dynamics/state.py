from uuid import uuid4

class Dimension:
    def __init__(self, name, value=0.0):
        self.id = uuid4()
        self.name = name
        self.value = value

class State:
    def __init__(self, dimensions: list[Dimension]):
        self.id = uuid4()
        self.dimensions = dimensions

class Action:
    def __init__(self, state: State, dimensions_effect: list[tuple[Dimension, float]]):
        self.id = uuid4()
        self.state = state
        #dimensions attribute is a tuple that contains a list of tuples (dimension object, effect value)
        self.dimensions_effect = dimensions_effect

    def __call__(self):
        for dim in self.dimensions_effect:
            dimension = dim[0]
            value = dim[1]
            dimension.value += value
        return "Effect applied"


if __name__ == "__main__":
    dim1 = Dimension("Benefits", value = 102000)
    dim2 = Dimension("EBITDA", value=80000)
    dim3 = Dimension("Taxes")
    dimensions_list = [dim1,dim2,dim3]
    tuple = [(dim1, -2000)]

    state = State(dimensions_list)
    
    for dim in state.dimensions:
        print(dim.value)

    Action(state,tuple)()

    for dim in state.dimensions:
        print(dim.value)
    

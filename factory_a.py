from abc import ABCMeta, abstractmethod

class IProduct(metaclass=ABCMeta):
    "A hypothetical class interface (product)"
    
    @staticmethod
    @abstractmethod
    def create_object():
        "An abstract interface method"
        pass
    
class ConcreteProductA(IProduct):
    "A Concrete class that implement the IProduct interface."
    
    def __init__(self):
        self.name = "ConcreteProductA"
        
    def create_object(self):
        return self

class ConcreteProductB(IProduct):
    "A Concrete class that implement the IProduct interface."
    
    def __init__(self):
        self.name = "ConcreteProductB"
        
    def create_object(self):
        return self

class ConcreteProductC(IProduct):
    "A Concrete class that implement the IProduct interface."
    
    def __init__(self):
        self.name = "ConcreteProductC"
        
    def create_object(self):
        return self

class FactoryA:
    " The factoryA class"
    
    @staticmethod
    def create_object(some_property):
        "A static method to get a concrete product."
        try:
            if some_property == 'a':
                return ConcreteProductA
            if some_property == 'b':
                return ConcreteProductB
            if some_property == 'c':
                return ConcreteProductC
            raise Exception('Class Not Found')
                
        except Exception as _e:
            print(_e)
        return None


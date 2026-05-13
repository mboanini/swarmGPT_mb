# This is a simple function that does nothing
def my_function(a, b, c):
    return d


# This class has indentation errors and uses a global variable without defining it
class BadClass:
    some_global_variable = global_variable  # Global variable is not defined

    def __init__(self):
        self.message = "This is a bad class"

    def print_message(self):
        print(self.mesage)  # Incorrect attribute name

from pathlib import Path

examples = [
    (
        "What is Python?",
        "Python is a programming language used for automation, "
        "web development, data analysis, and artificial intelligence."
    ),
    (
        "What is C++?",
        "C++ is a compiled programming language used for game engines, "
        "systems programming, and performance-critical applications."
    ),
    (
        "What is a variable?",
        "A variable is a name associated with a value in a program."
    ),
    (
        "How do I create a variable in Python?",
        "Assign a value to a name. Example: age = 25"
    ),
    (
        "How do I print text in Python?",
        'Use the print function. Example: print("Hello World")'
    ),
    (
        "How do I create a function in Python?",
        "Use the def keyword. Example: def greet(): print('Hello')"
    ),
    (
        "What is a Python list?",
        "A list stores an ordered collection of values."
    ),
    (
        "How do I create a Python list?",
        "Use square brackets. Example: numbers = [1, 2, 3]"
    ),
    (
        "What is a Python dictionary?",
        "A dictionary stores key-value pairs."
    ),
    (
        "How do I create a loop in Python?",
        "Use a for or while statement. Example: for i in range(5): print(i)"
    ),
    (
        "What is a conditional statement?",
        "A conditional statement executes code based on a Boolean condition."
    ),
    (
        "What is a class?",
        "A class defines attributes and methods used to create objects."
    ),
    (
        "What is an object?",
        "An object is an instance of a class."
    ),
    (
        "What is inheritance?",
        "Inheritance allows one class to extend another class."
    ),
    (
        "What is debugging?",
        "Debugging is the process of finding and fixing software errors."
    ),
    (
        "What is a syntax error?",
        "A syntax error occurs when code violates the grammar of a programming language."
    ),
    (
        "What is a runtime error?",
        "A runtime error occurs while a program is executing."
    ),
    (
        "What is an exception in Python?",
        "An exception is an event that interrupts normal program execution."
    ),
    (
        "How do I handle exceptions in Python?",
        "Use try and except blocks to handle expected errors."
    ),
    (
        "What is a pointer in C++?",
        "A pointer is a variable that stores a memory address."
    ),
    (
        "What is a reference in C++?",
        "A reference is an alias for an existing object."
    ),
    (
        "What is a vector in C++?",
        "A vector is a dynamically sized sequence container."
    ),
    (
        "What is the C++ standard library?",
        "The C++ standard library provides containers, algorithms, "
        "utilities, and input-output functionality."
    ),
    (
        "What is an algorithm?",
        "An algorithm is a sequence of instructions for solving a problem."
    ),
    (
        "What is a stack?",
        "A stack is a data structure that follows last-in, first-out ordering."
    ),
    (
        "What is a queue?",
        "A queue is a data structure that follows first-in, first-out ordering."
    ),
    (
        "What is a linked list?",
        "A linked list is a collection of nodes connected by references."
    ),
    (
        "What is binary search?",
        "Binary search finds a value in a sorted sequence by repeatedly "
        "reducing the search interval."
    ),
    (
        "What is recursion?",
        "Recursion occurs when a function calls itself to solve smaller problems."
    ),
    (
        "What is Big O notation?",
        "Big O notation describes how an algorithm's resource requirements "
        "grow with input size."
    ),
    (
        "What is Git?",
        "Git is a version control system used to track source code changes."
    ),
    (
        "What is a Git commit?",
        "A commit records a snapshot of changes in a Git repository."
    ),
    (
        "What is a Git branch?",
        "A branch is a movable reference to a line of development in Git."
    ),
    (
        "What is an API?",
        "An API defines how software components interact."
    ),
    (
        "What is REST?",
        "REST is an architectural style for designing networked applications."
    ),
    (
        "What is JSON?",
        "JSON is a text-based format for exchanging structured data."
    ),
    (
        "What is SQL?",
        "SQL is a language used to query and manage relational databases."
    ),
    (
        "What is a primary key?",
        "A primary key uniquely identifies each row in a relational database table."
    ),
    (
        "What is unit testing?",
        "Unit testing checks individual components of software for expected behavior."
    ),
    (
        "What is PyTorch?",
        "PyTorch is a machine learning framework that supports tensor computation "
        "and automatic differentiation."
    ),
    (
        "What is a tensor?",
        "A tensor is a multidimensional array used in numerical computation."
    ),
    (
        "What is a neural network?",
        "A neural network is a computational model with layers of learned parameters."
    ),
    (
        "What is a Transformer?",
        "A Transformer is a neural network architecture that uses attention "
        "to process sequences."
    ),
    (
        "What is backpropagation?",
        "Backpropagation computes gradients used to train neural networks."
    ),
    (
        "What is a learning rate?",
        "A learning rate controls the size of parameter updates during optimization."
    ),
    (
        "What is overfitting?",
        "Overfitting occurs when a model learns training data too closely "
        "and generalizes poorly."
    ),
    (
        "What is Unreal Engine?",
        "Unreal Engine is a game development engine supporting C++ and visual scripting."
    ),
    (
        "What is an Actor in Unreal Engine?",
        "An Actor is an object that can be placed or spawned in an Unreal Engine level."
    ),
    (
        "What is a component in Unreal Engine?",
        "A component adds reusable functionality or data to an Actor."
    ),
    (
        "What is Blueprint in Unreal Engine?",
        "Blueprint is Unreal Engine's node-based visual scripting system."
    ),
]

assert len(examples) == 50
assert len({question for question, _ in examples}) == 50

output = Path("programming_data_v1.txt")

text = "\n\n".join(
    f"User: {question}\nAssistant: {answer}"
    for question, answer in examples
) + "\n"

output.write_text(text, encoding="utf-8")

print(f"Created: {output}")
print(f"Unique programming conversations: {len(examples)}")
print(f"Dataset size: {output.stat().st_size} bytes")
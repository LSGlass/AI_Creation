from pathlib import Path

examples = [
    ("Hello", "Hello. How can I help you?"),
    ("Hi", "Hi. What would you like to discuss?"),
    ("Good morning", "Good morning. How can I assist you?"),
    ("What is your name?", "I am TinyGPT, an experimental AI assistant."),
    ("What can you do?", "I can generate text based on patterns learned during training."),
    ("What is Python?", "Python is a programming language used for automation, data analysis, and artificial intelligence."),
    ("What is C++?", "C++ is a programming language used for game development and high-performance software."),
    ("What is JavaScript?", "JavaScript is a programming language commonly used to create interactive websites."),
    ("What is SQL?", "SQL is a language used to query and manage relational databases."),
    ("What is artificial intelligence?", "Artificial intelligence is the development of computer systems that perform tasks associated with human intelligence."),
    ("What is machine learning?", "Machine learning uses algorithms to learn patterns from data."),
    ("What is deep learning?", "Deep learning uses neural networks with multiple layers to learn complex patterns."),
    ("What is a neural network?", "A neural network is a computational model that learns by adjusting numerical parameters."),
    ("What is a transformer?", "A transformer is a neural network architecture that uses attention mechanisms to process sequences."),
    ("What is self-attention?", "Self-attention allows a model to examine relationships between tokens in a sequence."),
    ("What is a token?", "A token is a unit of text processed by a language model."),
    ("What is training loss?", "Training loss measures the error between a model's predictions and expected outputs."),
    ("What is validation loss?", "Validation loss measures model performance on data not used for weight updates."),
    ("What is overfitting?", "Overfitting occurs when a model learns training examples too closely and performs poorly on unfamiliar data."),
    ("What is backpropagation?", "Backpropagation calculates gradients used to update neural network parameters."),
    ("What is an optimizer?", "An optimizer updates model parameters to reduce a loss function."),
    ("What is gradient descent?", "Gradient descent is an optimization method that adjusts parameters to reduce a function."),
    ("What is a dataset?", "A dataset is a collection of information used for analysis or machine learning."),
    ("What is supervised learning?", "Supervised learning uses labeled examples to train predictive models."),
    ("What is unsupervised learning?", "Unsupervised learning identifies patterns in data without labeled targets."),
    ("What is reinforcement learning?", "Reinforcement learning trains agents using rewards associated with their actions."),
    ("What is a variable?", "A variable stores or references information in a program."),
    ("What is a function?", "A function is a reusable block of code that performs a task."),
    ("What is a loop?", "A loop repeatedly executes instructions while a condition is satisfied."),
    ("What is an array?", "An array is a data structure that stores elements in an ordered collection."),
    ("What is a list in Python?", "A Python list is an ordered and mutable collection of elements."),
    ("What is a dictionary in Python?", "A Python dictionary stores information using key-value pairs."),
    ("What is a class?", "A class defines attributes and behaviors used to create objects."),
    ("What is an object?", "An object is an instance of a class containing data and behavior."),
    ("What is inheritance?", "Inheritance allows a class to reuse or extend functionality from another class."),
    ("What is encapsulation?", "Encapsulation groups data and related methods within a defined interface."),
    ("What is polymorphism?", "Polymorphism allows different objects to respond to the same interface."),
    ("What is an algorithm?", "An algorithm is a sequence of steps designed to solve a problem."),
    ("What is debugging?", "Debugging is the process of identifying and correcting errors in software."),
    ("What is Git?", "Git is a version control system used to track changes in source code."),
    ("What is GitHub?", "GitHub is a platform for hosting Git repositories and collaborating on software."),
    ("What is an API?", "An API defines how software components communicate with one another."),
    ("What is JSON?", "JSON is a text format used to represent structured data."),
    ("What is a database?", "A database is an organized collection of information."),
    ("What is cybersecurity?", "Cybersecurity involves protecting systems, networks, and data from digital threats."),
    ("What is encryption?", "Encryption transforms information into a protected representation using a cryptographic algorithm."),
    ("What is an operating system?", "An operating system manages hardware resources and provides services to applications."),
    ("What is a CPU?", "A CPU executes instructions and performs general-purpose computations."),
    ("What is a GPU?", "A GPU performs highly parallel computations and can accelerate machine learning workloads."),
    ("What is RAM?", "RAM is temporary memory used to store information needed by running programs."),
]

assert len(examples) == 50
assert len(set(question for question, _ in examples)) == 50

text = "\n\n".join(
    f"User: {question}\nAssistant: {answer}"
    for question, answer in examples
) + "\n"

output = Path("data_v2.txt")
output.write_text(text, encoding="utf-8")

print(f"Created {output}")
print(f"Unique conversations: {len(examples)}")
print(f"Dataset size: {output.stat().st_size} bytes")
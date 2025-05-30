def hello_world(name: str = "World") -> None:
    print(f"Hello, {name}!")
    print("This is a more complex hello world program.")
    print("It includes a function with a default argument and a return type hint.")

def greet_in_spanish(name: str) -> None:
    print(f"Hola, {name}!")

def greet_in_french(name: str) -> None:
    print(f"Bonjour, {name}!")

hello_world("Alice")
greet_in_spanish("Bob")
greet_in_french("Charlie")
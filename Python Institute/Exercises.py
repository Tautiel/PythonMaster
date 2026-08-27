class Cell:
    def __init__(self):
        self.food = 0
        
    def add_food(self, amount):
        self.food += amount
        
    def conseme_food(self, amount):
        consumed = min(self.food, amount)
        self.food -= consumed
        return consumed


class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def sit(self):
        print(f"{self.name} si siede")
        
    def roll_over(self):
        print(f"{self.name} si rotola ed è a pancia all'aria!")
        pass
    
my_dog = Dog("Willy", 6) 

print(f"Il mio cane si chiama {my_dog.name}.")
print(f"il mio cane ha {my_dog.age} anni.")

my_dog.sit()
my_dog.roll_over()

your_dog = Dog("Bruce", 5)

print(f"il tuo cane si chiama {your_dog.name}.")
print(f"il tuo cane ha {your_dog.age} anni.")

your_dog.roll_over()

class Restaurant:
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type
        
    def describe_restaurant(self):
        print(f"Ho il ristorante migliore del mondo che si chiama {self.restaurant_name}")
        print(f"Facciamo {self.cuisine_type}")
    
    def open_restaurant(self):
        print("Il ristorante è aperto H24")
        pass
    
my_restaurant = Restaurant("Italyesse", "italiana")

print(f"Il mio ristorante si chiama {my_restaurant.restaurant_name}")
print(f"Facciamo solo cunina {my_restaurant.cuisine_type}")

my_restaurant.describe_restaurant()
my_restaurant.open_restaurant()




    
       

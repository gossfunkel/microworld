import cell_life
from cell_life import libcell

if __name__ == '__main__':
    print(" = = =  TESTING cell_life library python wrapper  = = = ")

    test_cell = libcell.Cell_new(0, 1., 0)
    print(f"Cell sugars: {libcell.Cell_get_sugar(test_cell)}")
    libcell.Cell_add_sugar(test_cell, 2.)
    print(f"Sugar after adding 2: {libcell.Cell_get_sugar(test_cell)}")
    print(f"Cell water level: {libcell.Cell_get_water(test_cell)}")
    libcell.Cell_spend_water(test_cell, 3.)
    print(f"Water after spending 3: {libcell.Cell_get_water(test_cell)}")
    
    print(" = = =  testing concluded. goodbye!  = = = ")


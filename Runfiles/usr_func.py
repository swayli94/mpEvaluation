
import numpy as np


if __name__ == '__main__':
    
    dim_input = 2

    x = np.zeros(dim_input)
    
    with open('input.txt', 'r') as f:
        
        lines = f.readlines()

        for i in range(dim_input):
            x[i] = float(lines[i].split()[1])
    
    y = np.sum(x**2)
    
    with open('output.txt', 'w') as f:
        
        f.write(' y0  %.6f'%(y))
    
    
    
    
    
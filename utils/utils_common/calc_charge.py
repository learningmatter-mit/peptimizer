import numpy as np
import math

def net_charge(sequence):
    
    '''
    Utility function to calculate net charge of a sequence
    Reference: http://www.chem.ucalgary.ca/courses/351/Carey5th/Ch27/ch27-1-4-2.html

    Parameters
    ----------
    sequence:   str
                peptide sequence

    Returns
    -------
    charge: float
            net charge of sequence
    '''

    acidic = [sequence.count('D'), sequence.count('E'), sequence.count('C'), sequence.count('Y')]
    basic = [sequence.count('R'), sequence.count('K'), sequence.count('H')]

    acidic_pKa = [math.pow(10, 3.65), math.pow(10, 4.25), math.pow(10, 8.37), math.pow(10, 10.46)]
    basic_pKa = [math.pow(10, 10.76), math.pow(10, 9.74), math.pow(10, 7.59)]

    basic_coeff = [x*(1/(x+math.pow(10, 7))) for x in basic_pKa]
    acidic_coeff = [math.pow(10, 7)/(x+math.pow(10, 7)) for x in acidic_pKa]

    charge = - sum(np.multiply(acidic_coeff, acidic)) + sum(np.multiply(basic_coeff, basic))
    return charge
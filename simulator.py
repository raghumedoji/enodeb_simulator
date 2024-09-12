import numpy as np
import math
import matplotlib.pyplot as plt


# Function to compute propagation loss in dB
def calculate_oka_mora_prop_loss(d, freq, hB):
    PL = 69.55 + 26.16 * np.log10(float(freq)) - 13.82 * np.log10(float(hB)) + (44.9 - 6.55 * np.log10(float(hB))) * np.log10(float(d))
    return PL


#Function to compute shadowing loss in dB
def calculate_fading_loss():
    # scale is unit variance and 10 meters distance samples
    fading_loss_samples = np.random.rayleigh(scale=1, size=10)
    fading_loss_samples.sort()
    value = fading_loss_samples[-2]  # Second deepest fade
    fading_loss_in_db = 10 * np.log10(value)
    return fading_loss_in_db


# Function to compute Fading loss in dB
def calculate_shadowing_loss():
    mean = 0  # Mean of the log-normal distribution in dB
    std_dev = 2  # Standard deviation of the log-normal distribution in dB
    size = 500  # As step size is 10meters and road length5000m ,so 5000/10
    # Calculate the shadowing loss using a log-normal distribution with the given parameters
    shadowing_loss = np.random.lognormal(mean, std_dev, size)
    return shadowing_loss

def dot_product(v1, v2):
    # Calculate the dot product of two vectors
    return v1[0] * v2[0] + v1[1] * v2[1]

def vector_length(v):
    # Calculate the length of a vector
    return math.sqrt(dot_product(v, v))

# Function to calculate the angle from antenna vector and line from BS to user location
# and we know x3 = 20 for every point on the road
def calculate_user_angle(v1, v3):
    # Calculate the angle between two vectors(alpha BS vector and User location vector)
    dot_prod = dot_product(v1, v3)
    len_v1 = vector_length(v1)
    len_v3 = vector_length(v3)
    cos_theta = dot_prod / (len_v1 * len_v3)
    angle = np.arccos(cos_theta)
    angle = np.degrees(angle)
    return angle


# Function to calculate antenna discrimination
def antenna_discrimination_values(antenna_discrimination):
    # Open the text file for reading
    with open('/Users/chary/Downloads/antenna_pattern.txt', 'r') as f:
        # Loop through each line in the file
        for line in f:
            # Split the line into angle and angle discrimination
            angle, discrimination = line.strip().split()
            antenna_discrimination[round(float(angle))] = discrimination
    return antenna_discrimination


# function to run the simulation by taking user input
def run_simulation():
    #Below are the alpha and beta sectors
    alpha = 1
    beta = 0
    hB = 0.05
    # Location: 20 m west of the road at the midpoint of the road (e.g. at 2.5 km if the road is 5 km)
    # = 0.02
    # TX Power: PTX= 47 dBm
    PTX = 47
    # Line/connector Losses: L= 1 db
    line_loss = 1
    # Antenna Gain: GTX= 14.8 dBi (at boresight)
    GTX = 14.8
    freq = 800
    hm = 0.0015 #Height of the mobile in km
    k = -1
    #Vector of Basestation  alpha(x1,y1) are (0,1)
    v1 = np.array([0, 1])
    #vector of basestation  beta(x2,y2) are (root3/2, -1/2)
    v2 = np.array([np.sqrt(3)/2, -1/2])
    signal_level_alpha_y1 = []
    signal_level_beta_y2 = []
    signal_level_alpha_okamora_y4 = []
    signal_level_beta_okamora_y5 = []
    signal_level_alpha_okamora_shadowing_y6 = []
    signal_level_beta_okamora_shadowing_y7 = []
    signal_level_alpha_okamora_shadowing_fading_y8 = []
    signal_level_beta_okamora_shadowing_fading_y9 = []

    # Create an empty dictionary to store the angle discrimination data
    antenna_discrimination = {}
    antenna_discrimination = antenna_discrimination_values(antenna_discrimination)

    # Calculate fading loss only for 10meters distance
    PL_shadowing = calculate_shadowing_loss()

    # the road is divided into 1meter distance intervels
    for y in range(2500, -2501, -1):
        print('y value is', y)
        #Vector of user location at time = time
        v3 = np.array([20, y])
        #calculate distance from BS and BS is considered as (0,0)
        user_distance = vector_length(v3)
        # User distance in km
        user_distance = user_distance/1000
        print("User distance",user_distance)

        #calculate user angle considering BS as origin( angle b/w direct line from origin to user location
        # #and straight line joining origining and mid point of road
        user_angle = {}
        user_angle[(alpha, y)] = round(calculate_user_angle(v1, v3))
        print(user_angle[(alpha, y)])
        user_angle[(beta, y)] = round(calculate_user_angle(v2, v3))
        user_angle[(beta, y)] = round(user_angle[(beta, y)])
        print('user angle beta is', user_angle[(beta, y)])
        print('user angle alpha is', user_angle[(alpha, y)])

        #Check with Professor if angle beta is 360 - angle calculated?
        #if (0 < user_angle[(alpha, y)] < 120):
         #   user_angle[(beta, y)] = 120 - user_angle[(alpha, y)]

        print('Antenna discrimination for angle ', user_angle[(beta, y)], 'is', antenna_discrimination[user_angle[(beta, y)]])

        PL_okamora = calculate_oka_mora_prop_loss(user_distance, freq, hB)
        PL_okamora = round(PL_okamora)
        PL_Fading_alpha = calculate_fading_loss()
        PL_Fading_beta = calculate_fading_loss()
        PL_Fading_alpha = round(PL_Fading_alpha, 2)
        PL_Fading_beta = round(PL_Fading_beta, 2)

        EIRP_alpha = PTX -line_loss +GTX - float(antenna_discrimination[float(user_angle[(alpha, y)])])
        EIRP_beta = PTX -line_loss +GTX - float(antenna_discrimination[float(user_angle[(beta, y)])])
        print("The antenna discrimination for y for alpha ", y, "is", float(antenna_discrimination[float(user_angle[(alpha, y)])]))
        print("The antenna discrimination for y for beta ", y, "is", float(antenna_discrimination[float(user_angle[(beta, y)])]))

        print('User angle alpha', user_angle[(alpha, y)])
        print("The fading values are", PL_Fading_alpha, PL_Fading_beta)

        #Copy same PL shadowing for every 10meters distance on road
        if (y % 10 == 0):
            k=+1

        rsl_alpha = EIRP_alpha -PL_Fading_alpha -float(PL_okamora) +round(PL_shadowing[k], 2)
        rsl_beta = EIRP_beta -PL_Fading_beta -float(PL_okamora) +round(PL_shadowing[k])

        signal_level_alpha_y1.append(float(EIRP_alpha))
        signal_level_beta_y2.append(float(EIRP_beta))

        signal_level_alpha_okamora_y4.append(EIRP_alpha - float(PL_okamora))
        signal_level_beta_okamora_y5.append(EIRP_beta - float(PL_okamora))

        signal_level_alpha_okamora_shadowing_fading_y8.append(rsl_alpha)
        signal_level_beta_okamora_shadowing_fading_y9.append(rsl_beta)

        signal_level_alpha_okamora_shadowing_y6.append(EIRP_alpha + round(PL_shadowing[k], 2))
        signal_level_beta_okamora_shadowing_y7.append(EIRP_beta + round(PL_shadowing[k], 2))

    #Plot all 8 graphs here
    x = np.linspace(2500, -2500, 5001)

    #Figure1
    # Create a figure with two subplots
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    # Plot the first subplot
    ax1.plot(x, signal_level_alpha_y1, color='blue')
    ax1.set_title('EIRP Alpha Sector')
    ax1.set_xlabel('x is location of user on road')
    ax1.set_ylabel('Signal Level in dB')
    ax1.set_xlim(200, -200)
    # Plot the second subplot
    ax2.plot(x, signal_level_beta_y2, color='red')
    ax2.set_title('EIRP beta Sector')
    ax2.set_xlabel('x is location of user on road')
    ax2.set_ylabel('signal level in dB')
    ax2.set_xlim(200, -200)


    fig2, (ax1, ax2) = plt.subplots(1, 2)
    # Plot the first subplot
    ax1.plot(x, signal_level_alpha_okamora_y4, color='blue')
    ax1.set_title('EIRP Alpha Sector minus PLokamora')
    ax1.set_xlabel('x is location of user on road')
    ax1.set_ylabel('Signal Level in dB')
    ax1.set_xlim(200, -200)

    # Plot the second subplot
    ax2.plot(x, signal_level_beta_okamora_y5, color='red')
    ax2.set_title('EIRP beta Sector minus PLokamore')
    ax2.set_xlabel('x is location of user on road')
    ax2.set_ylabel('signal level in dB')
    ax2.set_xlim(200, -200)


    fig3, (ax1, ax2) = plt.subplots(1, 2)
    # Plot the first subplot
    ax1.plot(x, signal_level_alpha_okamora_shadowing_y6, color='blue')
    ax1.set_title('EIRP Alpha Sector minus PLokamora & PL shadowing')
    ax1.set_xlabel('x is location of user on road')
    ax1.set_ylabel('Signal Level in dB')
    ax1.set_xlim(200, -200)

    # Plot the second subplot
    ax2.plot(x, signal_level_beta_okamora_shadowing_y7, color='red')
    ax2.set_title('EIRP beta Sector minus PLokamore & PL shadowing')
    ax2.set_xlabel('x is location of user on road')
    ax2.set_ylabel('signal level in dB')
    ax2.set_xlim(200, -200)


    fig4, (ax1, ax2) = plt.subplots(1, 2)
    # Plot the first subplot
    ax1.plot(x, signal_level_alpha_okamora_shadowing_fading_y8, color='blue')
    ax1.set_title('EIRP Alpha Sector minus PLokamora & PL shadowing')
    ax1.set_xlabel('x is location of user on road')
    ax1.set_ylabel('Signal Level in dB')
    ax1.set_xlim(200, -200)

    # Plot the second subplot
    ax2.plot(x, signal_level_beta_okamora_shadowing_fading_y9, color='red')
    ax2.set_title('EIRP beta Sector minus PLokamore & PL shadowing')
    ax2.set_xlabel('x is location of user on road')
    ax2.set_ylabel('signal level in dB')
    ax2.set_xlim(200, -200)

    #Zoom In figures

    # Figure1
    # Create a figure with two subplots
    fig5, (ax1, ax2) = plt.subplots(1, 2)
    # Plot the first subplot
    ax1.plot(x, signal_level_alpha_y1, color='blue')
    ax1.set_title('EIRP Alpha Sector')
    ax1.set_xlabel('x is location of user on road')
    ax1.set_ylabel('Signal Level in dB')
    # Plot the second subplot
    ax2.plot(x, signal_level_beta_y2, color='red')
    ax2.set_title('EIRP beta Sector')
    ax2.set_xlabel('x is location of user on road')
    ax2.set_ylabel('signal level in dB')

    fig6, (ax1, ax2) = plt.subplots(1, 2)
    # Plot the first subplot
    ax1.plot(x, signal_level_alpha_okamora_y4, color='blue')
    ax1.set_title('EIRP Alpha Sector minus PLokamora')
    ax1.set_xlabel('x is location of user on road')
    ax1.set_ylabel('Signal Level in dB')
    # Plot the second subplot
    ax2.plot(x, signal_level_beta_okamora_y5, color='red')
    ax2.set_title('EIRP beta Sector minus PLokamore')
    ax2.set_xlabel('x is location of user on road')
    ax2.set_ylabel('signal level in dB')

    fig7, (ax1, ax2) = plt.subplots(1, 2)
    # Plot the first subplot
    ax1.plot(x, signal_level_alpha_okamora_shadowing_y6, color='blue')
    ax1.set_title('EIRP Alpha Sector minus PLokamora & PL shadowing')
    ax1.set_xlabel('x is location of user on road')
    ax1.set_ylabel('Signal Level in dB')
    # Plot the second subplot
    ax2.plot(x, signal_level_beta_okamora_shadowing_y7, color='red')
    ax2.set_title('EIRP beta Sector minus PLokamore & PL shadowing')
    ax2.set_xlabel('x is location of user on road')
    ax2.set_ylabel('signal level in dB')

    fig8, (ax1, ax2) = plt.subplots(1, 2)
    # Plot the first subplot
    ax1.plot(x, signal_level_alpha_okamora_shadowing_fading_y8, color='blue')
    ax1.set_title('EIRP Alpha Sector minus PLokamora & PL shadowing')
    ax1.set_xlabel('x is location of user on road')
    ax1.set_ylabel('Signal Level in dB')
    # Plot the second subplot
    ax2.plot(x, signal_level_beta_okamora_shadowing_fading_y9, color='red')
    ax2.set_title('EIRP beta Sector minus PLokamore & PL shadowing')
    ax2.set_xlabel('x is location of user on road')
    ax2.set_ylabel('signal level in dB')
    plt.show()


def main():
    run_simulation()


import matplotlib.pyplot as plt
import numpy as np

def generate_qam16_gray_figure():
    # 1. Define the Gray Code Mapping (PAM-4 for one axis)
    # We map 2 bits to the unnormalized amplitude levels.
    # Notice adjacent levels (e.g., -3 and -1) only differ by 1 bit ('00' vs '01').
    pam4_gray_map = [
        # Bits, Level
        ('00', -3),
        ('01', -1),
        ('11',  1),
        ('10',  3)
    ]

    # 2. Setup plot options
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # 3. Iterate through grid points to plot and label
    # The first 2 bits define the Real (I) axis position
    # The last 2 bits define the Imaginary (Q) axis position
    for bits_real, level_real in pam4_gray_map:
        for bits_imag, level_imag in pam4_gray_map:
            
            # Combine bits to form the 4-bit symbol code
            full_code = bits_real + bits_imag
            
            # Plot the constellation point (blue dot)
            # Using unnormalized coordinates for readability
            ax.plot(level_real, level_imag, 'bo', markersize=14, markeredgecolor='black')
            
            # Label the point with its 4-bit Gray code above the dot
            ax.text(level_real, level_imag + 0.4, full_code, 
                    ha='center', va='bottom', fontsize=14, fontweight='bold', color='darkblue')
            
            # (Optional) Label coordinates below the dot for clarity
            ax.text(level_real, level_imag - 0.4, f"({level_real},{level_imag})", 
                    ha='center', va='top', fontsize=10, color='gray')

    # 4. Final Formatting
    ax.set_title("Gray-coded 16-QAM Constellation\n(Unnormalized Coordinates)", fontsize=16)
    ax.set_xlabel("In-Phase (Real) Axis", fontsize=12)
    ax.set_ylabel("Quadrature (Imaginary) Axis", fontsize=12)
    
    # Draw center axes
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    
    # Set viewing limits and grid
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

    plt.tight_layout()
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    generate_qam16_gray_figure()
import matplotlib.pyplot as plt
import numpy as np

# Create a figure showing multiple examples for each defect type
plt.figure(figsize=(15, 10))

# Define the order in which to display classes with more descriptive labels
display_order = ["no anomaly", "cracking", "porosity", "lack of penetration"]

# Create a mapping from original dataset labels to new descriptive names
original_to_new = {
    "NoDifetto": "no anomaly",
    "Difetto1": "cracking",
    "Difetto2": "porosity",
    "Difetto4": "lack of penetration"
}

# Number of examples to show per category
num_examples = 3

# SPECIFY YOUR SELECTED INDICES FOR EACH CLASS HERE
# These specify the positions to use within each class's collection of images
selected_class_positions = {
    "no anomaly": [9, 2, 4],        # Use 1st, 16th, and 31st no_anomaly images
    "cracking": [0, 1, 2],           # Use 1st, 6th, and 11th cracking images  
    "porosity": [0, 2, 3],          # Use 1st, 11th, and 21st porosity images
    "lack of penetration": [7, 2, 3]  # Use 1st, 4th, and 7th lack_of_penetration images
}

# TRANSPOSED LAYOUT: anomaly types as columns, examples as rows
for col_idx, display_name in enumerate(display_order):
    # Find the original class name that maps to this display name
    original_class_name = next(orig for orig, new in original_to_new.items() if new == display_name)
    
    # Find index of this class in the dataset
    class_idx = class_names.index(original_class_name) if original_class_name in class_names else col_idx
    
    # Get all images of this class - these are the actual indices in the dataset
    class_images = np.where(full_train_labels == class_idx)[0]
    
    print(f"Found {len(class_images)} images for class '{display_name}'")
    
    # Select specific positions within this class's images
    example_indices = []
    
    # Get the positions specified for this class
    positions = selected_class_positions.get(display_name, [0, 1, 2])
    
    # Convert positions to actual dataset indices
    for pos in positions:
        if pos < len(class_images):
            # Get the actual dataset index at this position
            example_indices.append(class_images[pos])
        else:
            print(f"Warning: Position {pos} exceeds available images for {display_name}")
            # Use first image as fallback
            example_indices.append(class_images[0])
    
    # Ensure we have exactly num_examples
    while len(example_indices) < num_examples:
        example_indices.append(example_indices[0])
    
    # Show category name as a title for the column
    plt.figtext(0.125 + (col_idx * 0.25), 0.97, display_name, 
                ha='center', fontsize=10)
    
    # For each example of this category (now as rows)
    for row_idx, example_idx in enumerate(example_indices[:num_examples]):
        example_img = full_train_images[example_idx]
        
        # Calculate the subplot position (num_examples rows, 4 columns)
        plt_idx = row_idx * 4 + col_idx + 1
        
        plt.subplot(num_examples, 4, plt_idx)
        
        # Denormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        display_img = example_img * std + mean
        display_img = np.clip(display_img, 0, 1)
        
        # Display the image
        plt.imshow(display_img, cmap='gray')
        plt.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.94, bottom=0.05, hspace=0.025, wspace=0.1)
plt.show()

plt.savefig("weld_examples.png", dpi=600)


import polars as pl
import re

# Parse the raw data
raw_metrics = """
-----------------------------------------------------------------
 Epoch |   Train Loss |     Val Loss | Val Accuracy
-----------------------------------------------------------------
     1 |       1.1906 |       1.5886 |       0.2455
     2 |       1.0898 |       1.8105 |       0.2455
     3 |       1.0678 |       2.3924 |       0.2455
     4 |       0.9935 |       2.9915 |       0.2455
     5 |       0.9289 |       2.9536 |       0.2455
     6 |       0.9325 |       3.1001 |       0.2455
     7 |       0.8601 |       2.7732 |       0.2455
     8 |       0.8262 |       3.3698 |       0.2455
     9 |       0.7755 |       3.5766 |       0.2455
    10 |       0.6890 |       3.4601 |       0.2471
    11 |       0.6605 |       3.5591 |       0.2471
    12 |       0.5995 |       3.1489 |       0.3126
    13 |       0.5230 |       4.4606 |       0.3126
    14 |       0.4982 |       3.3592 |       0.3126
    15 |       0.3802 |       2.2891 |       0.2946
    16 |       0.3698 |       2.5085 |       0.2750
    17 |       0.3587 |       5.0025 |       0.2570
    18 |       0.2811 |       2.8534 |       0.3159
    19 |       0.3141 |      12.2290 |       0.3126
    20 |       0.1987 |       4.2747 |       0.3142
    21 |       0.1611 |       2.4542 |       0.3306
    22 |       0.1794 |       3.0853 |       0.2439
    23 |       0.1431 |       3.5357 |       0.1997
    24 |       0.1571 |       2.4904 |       0.2602
    25 |       0.1124 |       9.7362 |       0.2193
    26 |       0.1456 |      20.6819 |       0.1833
    27 |       0.1186 |       9.6439 |       0.2684
    28 |       0.0995 |      10.7408 |       0.1833
    29 |       0.0783 |       2.5213 |       0.3110
    30 |       0.0826 |       3.7614 |       0.3142
    31 |       0.0788 |      11.6001 |       0.2995
    32 |       0.0747 |       3.0095 |       0.3110
    33 |       0.0656 |       0.8967 |       0.6350
    34 |       0.0799 |       4.4573 |       0.2586
    35 |       0.0673 |       7.5838 |       0.2635
    36 |       0.0370 |       1.2496 |       0.5025
    37 |       0.0406 |       2.6823 |       0.3699
    38 |       0.0436 |      13.5701 |       0.2602
    39 |       0.0357 |       7.9930 |       0.2668
    40 |       0.0470 |       3.8594 |       0.4386
    41 |       0.0358 |       6.4262 |       0.3650
    42 |       0.0327 |       2.6316 |       0.3748
    43 |       0.0290 |      11.8909 |       0.2700
    44 |       0.0387 |       8.0534 |       0.2881
    45 |       0.0349 |       4.2531 |       0.2144
    46 |       0.0173 |       2.6802 |       0.3322
    47 |       0.0284 |       3.2264 |       0.3470
    48 |       0.0362 |       3.1173 |       0.3699
    49 |       0.0470 |       6.9349 |       0.3273
    50 |       0.0301 |       5.0401 |       0.2766
    51 |       0.0212 |       4.7269 |       0.3993
    52 |       0.0202 |       2.3598 |       0.4861
    53 |       0.0243 |       2.1298 |       0.4943
    54 |       0.0198 |       3.6179 |       0.3993
    55 |       0.0104 |       4.7796 |       0.3764
    56 |       0.0056 |       3.0391 |       0.4812
    57 |       0.0046 |       1.6318 |       0.6187
    58 |       0.0076 |       2.1635 |       0.6023
    59 |       0.0087 |       2.4738 |       0.6219
    60 |       0.0057 |       1.4831 |       0.6923
    61 |       0.0066 |       4.9311 |       0.4763
    62 |       0.0037 |       2.3433 |       0.6661
    63 |       0.0044 |       1.5562 |       0.7692
    64 |       0.0081 |       2.9634 |       0.6563
    65 |       0.0035 |       1.7737 |       0.7512
    66 |       0.0050 |       6.2404 |       0.4386
    67 |       0.0032 |       9.2946 |       0.3568
    68 |       0.0062 |       9.4672 |       0.3650
    69 |       0.0077 |       6.6005 |       0.4141
    70 |       0.0035 |       1.1239 |       0.6956
    71 |       0.0031 |       1.5988 |       0.6399
    72 |       0.0019 |       1.6513 |       0.6268
    73 |       0.0042 |       0.5123 |       0.8543
    74 |       0.0033 |       0.3987 |       0.9034
    75 |       0.0017 |       0.3713 |       0.9133
    76 |       0.0027 |       2.1158 |       0.5303
    77 |       0.0056 |       1.4728 |       0.6088
    78 |       0.0045 |       3.0423 |       0.4664
    79 |       0.0046 |       1.4365 |       0.6318
    80 |       0.0038 |       0.7918 |       0.7512
    81 |       0.0044 |       1.0431 |       0.7889
    82 |       0.0016 |       0.7263 |       0.8331
    83 |       0.0018 |       0.5024 |       0.8838
    84 |       0.0018 |       0.4843 |       0.8887
    85 |       0.0012 |       0.4649 |       0.8936
    86 |       0.0010 |       0.4676 |       0.9002
    87 |       0.0011 |       0.4331 |       0.8936
    88 |       0.0009 |       0.4097 |       0.9067
    89 |       0.0011 |       0.4167 |       0.8985
    90 |       0.0021 |       0.4068 |       0.9116
    91 |       0.0013 |       0.4066 |       0.9100
    92 |       0.0016 |       0.4362 |       0.9034
    93 |       0.0017 |       0.4080 |       0.9083
    94 |       0.0011 |       0.4148 |       0.9116
    95 |       0.0036 |       0.6242 |       0.8609
    96 |       0.0017 |       0.7211 |       0.8576
    97 |       0.0008 |       0.5486 |       0.8887
    98 |       0.0008 |       0.3991 |       0.9051
    99 |       0.0019 |       0.3919 |       0.9051
   100 |       0.0008 |       0.3911 |       0.9067
"""

# extract data using regular expressions
pattern = r'\s+(\d+)\s+\|\s+(\d+\.\d+)\s+\|\s+(\d+\.\d+)\s+\|\s+(\d+\.\d+)'
matches = re.findall(pattern, raw_metrics)

# lists for each column
epochs = []
train_losses = []
val_losses = []
val_accuracies = []

for match in matches:
    epoch, train_loss, val_loss, val_acc = match
    epochs.append(int(epoch))
    train_losses.append(float(train_loss))
    val_losses.append(float(val_loss))
    val_accuracies.append(float(val_acc))

# create the polars df
metrics_df = pl.DataFrame({
    "epoch": epochs,
    "train_loss": train_losses,
    "val_loss": val_losses,
    "val_accuracy": val_accuracies
})

print(metrics_df)

metrics_df.write_csv("training_metrics.csv")


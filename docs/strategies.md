# Initial Strategy
Use all 64 layers => ```64x512x512``` input and corresponding ```512x512```label

# 4 Layer Data Strategy
- Divide scan into 16 groups of 4 layers each (0..3, 4..7, ..., 60..63)
- Have dedicated label file for each group

Generate samples by taking first 4 images for one fragment (0..3) and using corresponding label file. Repeat for all groups.

=> ```4x512x512``` input and corresponding ```512x512```label


# 12 Layer Data Strategy
- One label file for 12 consecutive layers
- Pick the 12 best layers from the 64 layers based on hand-tuning
- Use an additional ignore mask to ignore certain areas

=> `12 x H x W` input + `H x W` label + `H x W` ignore mask
=> Pad input up to 16 layers for UNETR compatibility => `16 x H x W` input

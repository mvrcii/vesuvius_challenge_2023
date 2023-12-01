# New 4 layer data strategy

### Old approach:
Use all 64 layers => ```64x512x512``` input and corresponding ```512x512```label

### New approach
- Divide scan into 16 groups of 4 layers each (0..3, 4..7, ..., 60..63)
- Have dedicated label file for each group

Generate samples by taking first 4 images for one fragment (0..3) and using corresponding label file. Repeat for all groups.

=> ```4x512x512``` input and corresponding ```512x512```label
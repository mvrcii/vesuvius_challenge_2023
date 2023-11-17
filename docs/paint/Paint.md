# Anleitung
1  [Download and install Autohotkey](https://www.autohotkey.com/)

2 [Download and install newest Paint.net version](https://github.com/paintdotnet/release/releases/download/v5.0.11/paint.net.5.0.11.install.anycpu.web.zip)

3 Select all the layers you want to use from your scroll, drag them into Paint, select "Ebenen hinzufügen"

**DISCLAIMER** Don't add too many layers (RAM issue), ~ 25 worked for me with 16GB Ram, however this will also depend on the size of the image. Work around this issue by selecting only every 2nd layer e.g. 20, 22, ... 36, 40

4 In paint, add empty layer on top, set "Deckfähigkeit" to 70, select brush tool, choose some color, make sure **"Härte" at the top is set to 100%** (no soft borders)


5 Sort layers by dragging and dropping them in the right order, usually they are partially sorted from high to low already

6 Open the script ``paint_mrc.ahk`` with a text editor and change the limit in the first if statement to the number of images you added.

7 Start the script ``paint_mrc.ahk`` by double clicking it (the ahk extension should be associated with Autohotkey now)

8 Switch to paint
## Controls
W => Layer up

S => Layer down

A => Activate Eraser

D => Activate Brush

## DISCLAIMER 1, don't rotate the image, this crashes PAINT
## DISCLAIMER 2, don't manually enable/disable layers by clicking, this messes up the scripts count
## DISCLAIMER 3, don't use CTRL+Z to undo mistakes, use the eraser, as CTRL+Z undos layer visibility changes as well
(if you messed up the count, manually enable all layers visibility, then restart the script by double clicking it again and confirming to "replace the old instance")

To stop the script, expand the arrow at the bottom right of the task bar, right click the Autohotkey icon and select "Exit"

# Saving your result

To Save your result, go to the top layers, highlight everything, copy => New Image => Paste => Save as .png

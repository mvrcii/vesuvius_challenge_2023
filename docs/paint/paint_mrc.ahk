#IfWinActive, ahk_exe paintdotnet.exe

#NoEnv
SendMode Input
SetWorkingDir %A_ScriptDir%
counter := 1
delay := 15 ; Delay in milliseconds

; SET LIMIT IN IF STATETMENT BELOW TO NUMBER OF .tifs IN PAINT (not counting empty mask at top)
; Maps S to go down
s::
    if (counter < 25) {
        Loop, %counter%
        {
            Send, !{PgDn} ; Send Alt+Page Down (go to layer below)
            Sleep, %delay%
        }
        Send, ^{,} ; Send Ctrl+, (toggle layer visibility)
        Sleep, %delay%
        Send, ^!{PgUp} ; Send Ctrl+Alt+Page Up (go to top layer)
        counter++
    }

return

; Maps W to go UP
w::
    if (counter > 1)
    {
        Loop, % (counter-1)
        {
            Send, !{PgDn} ; Send Alt+Page Down (go to layer below)
            Sleep, %delay%
        }
        Send, ^{,} ; Send Ctrl+, (toggle layer visibility)
        Sleep, %delay%
        Send, ^!{PgUp} ; Send Ctrl+Alt+Page Up (go to top layer)
        counter--
    }
return

; Remap D to B to activate brush
d::b

; Remap A to E to activate eraser
a::e

#IfWinActive

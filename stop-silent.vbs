' stop-silent.vbs - klik dua kali untuk menghentikan semua layanan tanpa jendela.
Option Explicit
Dim sh, fso, dir
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.Run """" & dir & "\stop.bat"" /silent", 0, True
MsgBox "Layanan portal dan Data Mining UI sudah dihentikan.", vbInformation, "Portal Simulator dan ML"

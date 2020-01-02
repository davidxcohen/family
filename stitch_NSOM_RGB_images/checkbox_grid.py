from Tkinter import *

master = Tk()
master.title("Select Groups")

rows=10
columns=10

boxes = []
boxVars = []

# Create all IntVars, set to 0

for i in range(rows):
    boxVars.append([])
    for j in range(columns):
        boxVars[i].append(IntVar())
        boxVars[i][j].set(0)

def checkRow(i):
    global boxVars, boxes
    row = boxVars[i]
    deselected = []

    # Loop through row that was changed, check which items were not selected
    # (so that we know which indeces to disable in the event that 2 have been selected)

    for j in range(len(row)):
        if row[j].get() == 0:
            deselected.append(j)

    # Check if enough buttons have been selected. If so, disable the deselected indeces,
    # Otherwise set all of them to active (in case we have previously disabled them).

    if len(deselected) == (len(row) - 2):
        for j in deselected:
            boxes[i][j].config(state = DISABLED)
    else:
        for item in boxes[i]:
            item.config(state = NORMAL)

def getSelected():
    selected = {}
    for i in range(len(boxVars)):
        temp = []
        for j in range(len(boxVars[i])):
            if boxVars[i][j].get() == 1:
                temp.append(j + 1)
        if len(temp) > 1:
            selected[i + 1] = temp
    print(selected)


for x in range(rows):
    boxes.append([])
    for y in range(columns):
        Label(master, text= "Group %s"%(y+1)).grid(row=0,column=y+1)
        Label(master, text= "Test %s"%(x+1)).grid(row=x+1,column=0)
        boxes[x].append(Checkbutton(master, variable = boxVars[x][y], command = lambda x = x: checkRow(x)))
        boxes[x][y].grid(row=x+1, column=y+1)

b = Button(master, text = "Get", command = getSelected, width = 10)
b.grid(row = 12, column = 11)
mainloop()

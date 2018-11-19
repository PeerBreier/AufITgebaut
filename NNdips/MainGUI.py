# ======================
# imports
# ======================
from tkinter import messagebox
from tkinter import font
from tkinter import ttk
from tkinter import *
import Classes


# ================= Main Window =========================
class Application(Tk):
    """
    Main Application. Subclassed Tk().
    """
    def __init__(self):
        Tk.__init__(self)
        # todo: work on save and load functions
        # todo: implement function to change button state [disabled/normal]
        # ==== General Setup ====
        self.tk_setPalette(background='#FFFFFF')
        self.title("NNDips")
        self.iconbitmap(r'.\Images\Icons\TUM.ico')
        self.state('zoomed')
        self.protocol("WM_DELETE_WINDOW", self.ask_quit)
        self.minsize(800, 600)
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=14)
        style = ttk.Style(self)
        style.configure('Treeview', rowheight=32)
        self.drawGUI()  # draw GUI widgets and create GUI data container

    def drawGUI(self):
        """
        Creates child widgets for main application
        :return:
        """
        Classes.GUI(self),  # MAKE GuiData
        MenuBar(self),  # MAKE MENUBAR
        TabBar(self)  # MAKE TABBAR

    def reset(self):
        """
        Destroys all tk objects that are children of main application.
        :return:
        """
        for child in self.winfo_children():
            child.destroy()
        Classes.NNdipsCanv.plan = NONE
        self.drawGUI()

    def ask_quit(self):
        """
        Terminates Application session.
        :return:
        """
        if messagebox.askokcancel("Quit", "You are about to quit the Program. Unsaved data may be lost."):
            # app.ico = NONE  # prevents TypeErrors
            self.destroy()
            quit()


# Create TabBar some useful help FUNCTIONS NOT IMPLEMENTED
class TabBar(ttk.Notebook):
    """
    Creates TabBar and all parent level Frames in application layout. Then creates all child level widgets.
    """
    def __init__(self, root):
        ttk.Notebook.__init__(self, root)
        self.pack(expand=1, fill="both")

        upTab = Frame(root)
        self.add(upTab, text=' Upload ')
        wbTab = Frame(root)
        self.add(wbTab, text=' Workbench ')
        exTab = Frame(root)
        self.add(exTab, text=' Export ')

        # ==== Upload Tab ====
        uplCanvFrame = Frame(upTab)
        uplCanvFrame.place(anchor=NW, relheight=.9, relwidth=1)
        Classes.NNdipsCanv(uplCanvFrame, 'upPlan').pack(fill=BOTH, padx=1, pady=1, expand=True)

        uplBtnFrame = Frame(upTab, bd=4)
        uplBtnFrame.place(anchor=NW, relheight=.1, relwidth=1, rely=.9)
        Classes.UploadTools(uplBtnFrame)

        # ==== Workbench Tab ====
        wbCanvFrame = Frame(wbTab)
        wbCanvFrame.place(anchor=NW, relheight=.9, relwidth=.7)
        Classes.NNdipsCanv(wbCanvFrame, 'wbPlan').pack(fill=BOTH, padx=1, pady=1, expand=True)

        wbResCanvFrame = Frame(wbTab)
        wbResCanvFrame.place(anchor=NW, relheight=.5, relwidth=.3, relx=.7)
        Classes.NNdipsCanv(wbResCanvFrame, 'wbResult', 'result').pack(fill=BOTH, padx=1, pady=1, expand=True)

        wbPlanListFrame = Frame(wbTab, bd=1)
        wbPlanListFrame.place(anchor=NW, relheight=.2, relwidth=.3, relx=.7, rely=.5)
        col = [["Plan Nr.", "scanned", "Hits"], [150, 100, 120]]
        Classes.PlanListBox(wbPlanListFrame, col, 'wbPlan')

        wbResToolsFrame = Frame(wbTab, bd=4)
        wbResToolsFrame.place(anchor=NW, relheight=.1, relwidth=.7, rely=.9)
        Classes.ResultTools(wbResToolsFrame)

        wbResListFrame = Frame(wbTab, bd=1)
        wbResListFrame.place(anchor=NW, relheight=.2, relwidth=.3, relx=.7, rely=.7)
        col = [["Hits", "x-Coord", "y-Coord", "Comment"], [20, 50, 50, 100]]
        Classes.ResultListBox(wbResListFrame, col, 'wbResult')

        wbCommentFrame = Frame(wbTab, bd=4)
        wbCommentFrame.place(anchor=NW, relheight=.1, relwidth=.3, relx=.7, rely=.9)
        Classes.ResComment(wbCommentFrame)

        # ==== Tab 3 - Results ====
        exSymListFrame = Frame(exTab, bd=1)
        exSymListFrame.place(anchor=NW, relheight=1, relwidth=.2, relx=.6)
        col = [["Hits"], [20]]
        Classes.ExSymListBox(exSymListFrame, col, 'exSymbol')

        exPlanListFrame = Frame(exTab)
        exPlanListFrame.place(anchor=NW, relheight=1, relwidth=.2, relx=.8)
        col = [["Plan Nr.", "scanned"], [150, 100]]
        Classes.ExPlanListBox(exPlanListFrame, col, 'exPlan')

        Classes.ExTools(exTab, bd=5).place(anchor=NW, relheight=1, relwidth=.6)


class MenuBar(Menu):
    """
    Creates MenuBar for main application.
    """
    def __init__(self, root):
        # ==== Creating a Menu Bar ====
        Menu.__init__(self, root)
        root.config(menu=self)

        # ==== Add File menu ====
        fileMenu = Menu(self)
        self.add_cascade(label="File", menu=fileMenu)
        fileMenu.add_command(label="New Project", command=lambda: root.reset())
        fileMenu.add_command(label="Add CNN", command=lambda: Classes.addCNNsymbol.initialize())
        fileMenu.add_command(label="Open Project", command=lambda: Classes.GUI.data.loadData())
        fileMenu.add_command(label="Save Project", command=lambda: Classes.GUI.data.saveData())
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=root.ask_quit)

        # ==== Add Help menu ====
        helpMenu = Menu(self)
        self.add_cascade(label="Help", menu=helpMenu)
        helpMenu.add_command(label="About", command=Classes.AboutBox)


class Ico:
    """
    Stores all icon sized images for buttons and other widgets.
    """
    def __init__(self, symDirs):
        self.arrowLeft = PhotoImage(file=r'.\Images\Icons\Backward_64x.png')
        self.arrowRight = PhotoImage(file=r'.\Images\Icons\Forward_64x.png')
        self.zoomToFit = PhotoImage(file=r'.\Images\Icons\ZoomToFit_64x.png')
        self.symIco = []
        self.renewSymIco(symDirs)

    def renewSymIco(self, symDirs):
        self.symIco = []
        for sym in symDirs:
            symIPath = './CNNs/' + sym + '/Icon.png'
            self.symIco.append(PhotoImage(file=symIPath))


# =========================================
# Start GUI
# ======================
if __name__ == '__main__':  # you always need that
    app = Application()
    app.mainloop()

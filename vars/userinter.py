from vars.utilities import util


class menu():
    def __init__(self) -> None:
        print("address?")
        self.address = str(input())
        self.ut = util(self.address)
        print("auto/manual?")
        val = str(input()).capitalize()
        if val == "M":
            self.manual()
        elif val == "A":
            pass
        elif val == "X":
            pass
        else:
            print()

    def manual(self):
        self.ut.image_load()
        self.ut.image_visulazation()
        self.ut.segment_img()
        self.ut.segment_selection(img_no=8)
        self.ut.seperate_segment()
        self.ut.msng_pxl_flr()
        self.ut.ff11()

    def automatic():
        pass

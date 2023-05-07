from data.rico import Rico
from data.publaynet import PubLayNet
from data.magazine import Magazine
from data.ricoTest import RicoTest

def get_dataset(name, split, transform=None):
    if name == 'rico':
        # tmp = Rico(split, transform)
        # tmp.process()
        return Rico(split, transform)#by ljw 20230131

    elif name == 'publaynet':
        return PubLayNet(split, transform)

    elif name == 'magazine':
        return Magazine(split, transform)

    #by ljw 20221102
    #加入ricoTest
    elif name == 'ricoTest':
        # tmp = RicoTest(split, transform)
        # tmp.process()
        return RicoTest(split, transform)


    raise NotImplementedError(name)

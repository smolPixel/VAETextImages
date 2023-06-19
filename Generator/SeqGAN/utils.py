

class Signal:
    """Running signal to control training process"""

    def __init__(self):
        self.pre_sig = True
        self.adv_sig = True

        self.update()

    def update(self):
        signal_dict = {'pre_sig': True, 'adv_sig': True}
        self.pre_sig = signal_dict['pre_sig']
        self.adv_sig = signal_dict['adv_sig']
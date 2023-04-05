from taming.data.faceshq import FFHQTrain, FFHQValidation

class FFHQTrainRDM(FFHQTrain):
    def __init__(self,size,*args,**kwargs):
        super().__init__(size=size,*args,**kwargs)
        self.size = size


class FFHQValidationRDM(FFHQValidation):
    def __init__(self, size, *args, **kwargs):
        super().__init__(size=size, *args, **kwargs)
        self.size = size

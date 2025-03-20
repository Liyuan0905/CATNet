from model.swin.swin import Swin
from model.swin.swin import EBM_Prior

from model.swin.swin import PureDecoder

def get_model(option):
    model_name = option['model_name']
    
    pure_model = PureDecoder(option['trainsize'])
    proxy_model = PureDecoder(option['trainsize'])
    
    
    print("Pure_Model based on {} have {:.4f}Mb paramerters in total".format(model_name, sum(x.numel()/1e6 for x in pure_model.parameters())))
    
    
    
    return pure_model.cuda(), proxy_model.cuda() 
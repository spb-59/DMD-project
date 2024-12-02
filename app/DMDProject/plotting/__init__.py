
import mpld3
from matplotlib import pyplot as plt
import pickle
import logging


def make_plots(data):

    plots=[]
    for frame in data:
        logging.info(frame['frameID'])
        leads=[]
        plt.switch_backend('Agg')
        plt.axis('off')
        for lead in pickle.loads(frame['frameData']).T:
            
            fig, ax = plt.subplots()
            plt.axis('off')
            ax.plot(lead,label='ECG signal')
            ax.set_title('Signal')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.axis('off')
            ax.set_xticks([0, 500, 1000, 1500, 2000])  
             
            ax.set_yticks([])
   

            html= mpld3.fig_to_html(fig)
            plt.close()
            leads.append(html)
        
        plots.append(leads)
    
    logging.info(len(plots[0]))
    return plots

def make_metric(data):
    control = {
    "R_N": 0.122651,
    "R_L": 0.123237,
    "R_M": 0.249697,
    "R_P": 0.224652,
    "Lam_min": -1.000000,
    "Lam_max": 0.996668,
    "M_u": 0.000007,
    "M_s": 0.000004,
    "P_u": 0.004248,
    "P_s": 0.004099,
}
    features=["R_N", 'R_L', 'R_M', 'R_P', 'Lam_min', 'Lam_max', 'M_u', 'M_s', 'P_u', 'P_s']
    avg={}


    for point in data:
        for f in features:
            avg[f]=avg.get(f,0)+point[f]
    
    lenData = len(data)
    for f in avg:
        avg[f] = avg[f] / lenData

    logging.info(avg)
    


    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(6, 4)) 


    x_positions = list(range(10)) 
    ax.bar(x_positions, [avg[f] for f in features], width=0.4, label="Patient Data", align='center')
    ax.bar(x_positions, [control[f] for f in features], width=0.4, label="Control Values", align='edge')


    
    

    ax.set_xticklabels(["R_N", 'R_L', 'R_M', 'R_P', 'Lam_min', 'Lam_max', 'M_u', 'M_s', 'P_u', 'P_s'],rotation=45,ha='right')

    ax.set_title('Current vs Control Values')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Value')

    
    ax.legend()
    plt.tight_layout()

    
    html = mpld3.fig_to_html(fig)
    plt.close()

    return html





        
        



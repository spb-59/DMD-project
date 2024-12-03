/* eslint-disable react/prop-types */
import { useState,useEffect } from "react";
import { ArrowRight,ArrowLeft } from "lucide-react";
const API_BASE = import.meta.env.VITE_API_URI
const CODE='?code='+import.meta.env.VITE_FUNCTION_KEY


export default function ECGSig({plots,prediction,comments,date,id}) {
    const leads=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    const predictions=['Healthy','Atrial Fibrillation','Myocardial Infarction']

    const [frame,setFrame]=useState(0)
    const [lead,setLead]=useState(0)
    const[comment,setComment]=useState(comments[frame][0])
    const [save,setSave]=useState(true)
    useEffect(() => {
        
        setComment(comments[frame][0]);

        console.log(comment)
      // eslint-disable-next-line react-hooks/exhaustive-deps
      }, [frame]);
    const handleSave = async () => {
        try {
          const response = await fetch(API_BASE+'/update_frame_comment'+CODE, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              comment,
              resultID:comments[frame][1]
            }),
          });
    
          if (response.ok) {
            setSave(true)
          } else {
            setSave(false)
          }
          comments[frame][0]=comment
        } catch (error) {
            setSave(false)
          console.error('Error saving comment:', error);
        }
      };
    
    if (!plots) return;

  return(
    <div className="flex items-center ">
    <div className="w-1/2 flex flex-col items-center justify-center border-r border-black">
    <div className="flex gap-4 mt-10">

    <ArrowLeft color={frame>0?'black':'gray'} onClick={()=>{if (frame>0) setFrame(frame-1)}}/> Frame {frame+1} / {plots.length} <ArrowRight color={frame+1<plots.length?'black':'gray'} onClick={()=>{if (frame+1<plots.length) setFrame(frame+1)}} />
    </div>
    <div className=" flex gap-2">
        {leads.map((_lead,index)=>(
          <span key={index} className={`border-l hover:cursor-pointer ${index==0?'border-none':' '} border-black px-2 py-1 ${index==lead ?'font-bold':' '}`} onClick={()=>{setLead(index)}}>
            {_lead}
          </span>
        ))}
    </div>
    <iframe
          title="mpld3 Plot"
          src={`data:text/html;charset=utf-8,${encodeURIComponent(plots[frame][lead])}`}
          width="90%"
          height="500px"
/>

    </div>
    <div className='flex w-1/2 justify-center items-center'>
        <div className={`w-3/4 h-3/4 flex flex-col border-2 p-4 gap-2 ${prediction[frame]==0?'border-green-600':'border-red-600'} `}>
          <span className="border-b"><span className="font-bold">Frame Number:</span> {frame+1}</span>
          <span className="border-b"><span className="font-bold">Frame ID:</span> {comments[frame][1]}</span>


          <span className="border-b"><span className="font-bold">Record Date:</span> {date}</span>

          <span className="border-b"><span className="font-bold">RecordID:</span> {id}</span>
          <span className={`border-b  ${prediction[frame]==0?' ':'text-red-600'}`}><span className={`font-bold  `} >Prediction: </span>{predictions[prediction[frame]]} </span>

          <span className="flex flex-col gap-1 "><span className="font-bold">Frame Comments:</span>
          <textarea className="border h-40 p-1"  key={frame}   value={comment} 
        onChange={(e) => setComment(e.target.value)} ></textarea>
          <button className="p-1 border-black border w-1/4 self-end rounded-lg" onClick={handleSave}>Save {!save?'Failed':''}</button>
           </span>

        </div>
      </div>
    
    </div>
  );
}

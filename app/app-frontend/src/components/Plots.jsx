/* eslint-disable react/prop-types */
import { useEffect, useState } from "react";
import 'd3' 
import 'mpld3' 
import {ECGSig} from ".";
import { ArrowLeft } from "lucide-react";

const API_BASE = import.meta.env.VITE_API_URI


export default function Plots({ id,setBack }) {
 
  const [plots, setPlots] = useState(null);
  const [metric, setMetric] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [comments, setComments] = useState(null);
  const[record,setRecord]=useState(false)
  const [comment,setComment]=useState('')
const[save,setSave]=useState(true)
  const handleSave = async () => {
    try {
      const response = await fetch(API_BASE+'/update_record_comment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          comment,
          recordID:id
        }),
      });
      if (response.ok) {
        setSave(true)
      } else {
        setSave(false)
      }
    } catch (error) {
        setSave(false)
      console.error('Error saving comment:', error);
    }
  };

  useEffect(() => {
    fetch(API_BASE+`/get_metric?record_id=${id}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })
    .then(response => response.json())  
    .then(data => {
      setPlots(data.plots);
      setMetric(data.metric);
      setPredictions(data.predictions);
      setComments(data.comments);
      setRecord(data.record)
    })
    .catch(error => {
      console.error('Error fetching data:', error);
    });

  }, [id]);

useEffect(()=>{
  setComment(record[1])
},[record])
if(!plots||!metric) {return( <>
  Loading ...
</>)}

const _risk=['Low','Medium','High','Extreme']
const colors=['green-600','yellow-600','orange-600','red-600']
const sum = Math.min(predictions.reduce((sum, num) => num !== 0 ? sum + num + 1 : sum, 0)-1,_risk.length-1)
const risk = _risk[sum];
const DOB=new Date(record[3])
const age = new Date().getFullYear() - DOB.getFullYear();

  return (
    <>
    <div className="flex items-start justify-start p-4">  <div className="w-[5%] flex hover:cursor-pointer " onClick={()=>setBack(false)}>
     <ArrowLeft />Back </div>  <div className="text-[60px] pt-4 "> Record Summary</div>
    </div>

    <div className="flex w-[100vw] items-center justify-center">
      
      <div className='flex w-1/2 justify-center items-center h-full border-r border-black'>
      <div className={`w-3/4 h-3/4 flex flex-col border-2 p-4 gap-2 border-${colors[sum]}`}>
          <span className="border-b"><span className="font-bold">Patient Name:</span> {record[4]}</span>
          <span className="border-b"><span className="font-bold">PatientID :</span> {id}</span>
          <span className="border-b"><span className="font-bold">Age :</span> {age}</span>
          
          <span className="border-b"><span className="font-bold">Record Date:</span> {record[2]}</span>
          <span className={`border-b text-${colors[sum]}`}><span className="font-bold">Severity:</span> {risk}</span>
          <span className=""><span className="font-bold">Record ID:</span> {record[0]}</span>
          <span className="flex flex-col gap-1 "><span className="font-bold">Record Comments:</span>
          <textarea className="border h-40 p-1"   value={comment}
        onChange={(e) => setComment(e.target.value)} ></textarea>
          <button className="p-1 border-black border w-1/4 self-end rounded-lg" onClick={handleSave}>Save {!save?'Failed':''} </button>
           </span>
        </div>
      </div>
      <div className='flex w-1/2  items-center justify-start'>
        <iframe
          title="mpld3 Plot"
          src={`data:text/html;charset=utf-8,${encodeURIComponent(metric)}`}
          width="90%"
          height="420px" />
      </div>
    </div><div>
        <ECGSig plots={plots} prediction={predictions} comments={comments} date={record[2]} id={record[0]}/>
      </div></>
  );
}

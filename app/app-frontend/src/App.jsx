import { useState } from "react";
import { Search, Upload } from "./components";


export default function App() {
  const [upload,setUpload]=useState(false)
  const [search,setSearch]=useState(false)
  // const [about, setAbout]=useState(false)

  if(upload){
    return <Upload setUpload={setUpload}/>
  }
  if (search){
    return <Search setSearch={setSearch} />
  }

  return(
<main className="flex flex-col items-center justify-center min-h-[100vh]">
<img src="/blue2.jpg" className="  h-auto  w-full absolute -z-50 opacity-50" />
<h1 className="font-bold text-[72px] text-center relative ">
  
  ECG prediction system
</h1>
<div className="flex gap-2 items-center justify-center">
<button className="p-4 border-2 rounded-lg border-black z-50 bg-white " onClick={()=>setSearch(true)}>
Search Record
 </button>
<button className="p-4 border-2 rounded-lg border-black z-50 bg-white " onClick={()=>setUpload(true)}>
New Record
</button>
{/* <button className="p-4 border-2 border-black " onClick={()=>setAbout(true)}>
About
 </button> */}
</div>

</main>
  );
}

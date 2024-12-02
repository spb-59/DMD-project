import { ArrowLeft, SearchIcon } from "lucide-react";
import { useState } from "react";
import Plots from "./Plots";

const API_BASE = import.meta.env.VITE_API_URI
const CODE='?code='+import.meta.env.VITE_FUNCTION_KEY


// eslint-disable-next-line react/prop-types
export default function Search({ setSearch }) {
  const [query, setQuery] = useState("");
  const [plot, setPlot] = useState(false);
  const [item, setItem] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSearch = async (event) => {
    event.preventDefault()
    if (query.trim() === "") return; 
    setLoading(true);
    setError(null); 

    try {

      const response = await fetch(API_BASE+`/get_search`+CODE+`&q=${query}`);
      const data = await response.json();


      setResults(data.data || []);
    } catch (err) {
      setError("An error occurred while fetching data.");
      console.error(err)
    } finally {
      setLoading(false);
    }
  };
  const getPlots=(id)=>{
    setPlot(true)
    setItem(id)
  }

  if(plot){
    return(
    <Plots id={item} setBack={setPlot} />
  )}

  return (
    <>
      <div className="flex items-start justify-start p-4">
        <div
          className="w-[5%] flex hover:cursor-pointer"
          onClick={() => setSearch(false)}
        >
          <ArrowLeft /> Back
        </div>
        <div className="text-[60px] pt-4">Search Record</div>
      </div>

      <div className="flex justify-center items-center">
      <form
  onSubmit={handleSearch} // Use handleSearch here
  className="flex w-3/4 border-2 border-gray-300 rounded-xl p-1 bg-gray-50 hover:bg-gray-100"
>
  <input
    type="text"
    placeholder="Search by Name, RecordID or PatientID"
    value={query}
    onChange={handleInputChange}
    className="flex flex-col items-center justify-center w-full h-10 p-2 cursor-pointer"
  />
  <button className="p-2" type="submit"> {/* Type set to "submit" */}
    {loading ? <span>Loading...</span> : <SearchIcon />}
  </button>
</form>
      </div>

      {error && <p className="text-red-500 text-center mt-4">{error}</p>}

      {results.length > 0 && (
        
          <div className="flex flex-col gap-2 items-center justify-center mt-2">
            {results.map((item, index) => (
              <div key={index} className={`w-3/4 flex border-2 p-4 gap-2 shadow-md hover:cursor-pointer`} onClick={()=>{getPlots(item.recordID)}}>
                <div className="w-1/2 flex flex-col">
              <span className="border-b"><span className="font-bold">Patient Name:</span> {item.name}</span>
              <span className="border-b"><span className="font-bold">Patient ID :</span> {item.patientID}</span>
              <span className=""><span className="font-bold">Record ID :</span> {item.recordID}</span>
              </div>
              <div className="w-1/2 flex flex-col">
              <span className="border-b"><span className="font-bold">DOB:</span> {item.DOB}</span>
              <span className="border-b"><span className="font-bold">Date :</span> {item.date}</span>
              </div>
              </div>
            ))}
          </div>
      )}
    </>
  );
}

/* eslint-disable react/prop-types */
import { useState } from 'react';
import { Plots } from '.';
import axios from 'axios';
import { ArrowLeft } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URI
const CODE='?code='+import.meta.env.VITE_FUNCTION_KEY

const Upload = ({ setUpload }) => {
  const [signalFile, setSignalFile] = useState(null);
  const [headerFile, setHeaderFile] = useState(null);
  const [patientID, setPatientID] = useState(''); // State to store Patient ID
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [newPateint,setPatient]=useState(false)
  const [patientName, setPatientName] = useState('');
  const [patientDOB, setPatientDOB] = useState('');
  const [patientContact, setPatientContact] = useState('');

  const handleFileChange = (event) => {
    const files = event.target.files;
    if (files.length !== 2) {
      alert('You must select exactly two files!');
      return;
    }

    const matFile = Array.from(files).find((file) => file.name.endsWith('.dat'));
    const heaFile = Array.from(files).find((file) => file.name.endsWith('.hea'));

    if (matFile && heaFile) {
      const matBaseName = matFile.name.split('.').slice(0, -1).join('.');
      const heaBaseName = heaFile.name.split('.').slice(0, -1).join('.');

      if (matBaseName === heaBaseName) {
        setSignalFile(matFile);
        setHeaderFile(heaFile);
      } else {
        alert('The .dat and .hea files must have the same base name!');
      }
    } else {
      alert('Please select one .dat file and one .hea file!');
    }
  };

  const getID=async()=>{
    const payload = {
        name: patientName,   
        DOB: patientDOB,    
        contact: patientContact, 
      };
      
      try {
        setLoading(true);
        const res = await axios.post(
          API_BASE+'/create_patient'+CODE, 
          payload, 
          {
            headers: {
              'Content-Type': 'application/json',
            },
          }
        );
        const data=res.data;
        console.log(data.patientID)
        return data.patientID
      } catch (error) {
        console.error(error);


  }}
  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!signalFile || !headerFile) {
      alert('Both files are required!');
      return;
    }
    let id=''
    if (newPateint) {
         id = await getID();
     }
     else{
        id=patientID
     }
    if (!id.trim()) {
      alert('Patient ID is required!');
      return;
    }

    const formData = new FormData();
    formData.append('signal', signalFile);
    formData.append('header', headerFile);
    formData.append('patientID', id); 

    try {
      setLoading(true);
      const res = await axios.post(
        API_BASE+'/insert_record'+CODE,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      setResponse(res.data);
    } catch (error) {
      console.error(error);
      setResponse({ status: 'error', message: 'File upload failed.' });
    } finally {
      setLoading(false);
    }
  };

  if (response) {
    return (
      <div className="w-[100vw] h-[100vh]">
        <Plots id={response.recordID} setBack={setUpload} />
      </div>
    );
  }

  return (
    <><div className="flex items-start justify-start p-4">  <div className="w-[5%] flex hover:cursor-pointer " onClick={() => setUpload(false)}>
          <ArrowLeft />Back </div>  <div className="text-[60px] pt-4 "> Create record </div>
      </div><div className="flex flex-col items-center justify-center p-8 ">


              <form onSubmit={handleSubmit} className="flex flex-col w-1/4">
                  {!newPateint && <label className="font-semibold ">
                      Patient ID:
                      <input
                          type="text"
                          value={patientID}
                          onChange={(e) => setPatientID(e.target.value)} 
                          className="flex flex-col items-center justify-center w-full h-8 border-2 border-gray-300 rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100" />
                  </label>}
                  {newPateint && <>
                      <label className="font-semibold mb-2">
                          Patient Name:
                          <input
                              type="text"
                              value={patientName}
                              onChange={(e) => setPatientName(e.target.value)}
                              className="flex items-center justify-center w-full h-8 border-2 border-gray-300 rounded-lg bg-gray-50 hover:bg-gray-100" />
                      </label>
                      <label className="font-semibold mb-2">
                          Patient DOB:
                          <input
                              type="date"
                              value={patientDOB}
                              onChange={(e) => setPatientDOB(e.target.value)}
                              className="flex items-center justify-center w-full h-8 border-2 border-gray-300 rounded-lg bg-gray-50 hover:bg-gray-100" />
                      </label>
                      <label className="font-semibold mb-4">
                          Patient Contact:
                          <input
                              type="text"
                              value={patientContact}
                              onChange={(e) => setPatientContact(e.target.value)}
                              className="flex items-center justify-center w-full h-8 border-2 border-gray-300 rounded-lg bg-gray-50 hover:bg-gray-100" />
                      </label>
                  </>}
                  <div className="flex flex-col items-center justify-center w-1/2 h-8 border-2 border-gray-300 rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 mt-4" onClick={() => setPatient(!newPateint)}> {!newPateint ? 'New Patient' : 'Existing patient'}</div>
                  <div
                      className="flex flex-col items-center justify-center w-full mb-4 mt-4"
                      onDrop={(e) => handleFileChange(e)}
                      onDragOver={(e) => e.preventDefault()}
                  >
                      <label
                          htmlFor="file-input"
                          className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100"
                      >
                          <div className="flex flex-col items-center justify-center pt-5 pb-6">
                              {signalFile && (
                                  <div className="flex flex-col space-y-2">
                                      <p className="text-sm text-green-600">Signal File: {signalFile.name}</p>
                                      <p className="text-sm text-green-600">Header File: {headerFile.name}</p>
                                  </div>
                              )}
                              {!signalFile && (
                                  <>
                                      <svg
                                          className="w-8 h-8 mb-4 text-gray-500"
                                          aria-hidden="true"
                                          xmlns="http://www.w3.org/2000/svg"
                                          fill="none"
                                          viewBox="0 0 20 16"
                                      >
                                          <path
                                              stroke="currentColor"
                                              strokeLinecap="round"
                                              strokeLinejoin="round"
                                              strokeWidth="2"
                                              d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2" />
                                      </svg>
                                      <p className="mb-2 text-sm text-gray-500">
                                          <span className="font-semibold">Click to upload</span>
                                      </p>
                                      <p className="text-xs text-gray-500">.dat and .hea files</p>
                                  </>
                              )}
                          </div>
                          <input
                              id="file-input"
                              type="file"
                              className="hidden"
                              accept=".dat,.hea"
                              multiple
                              onChange={handleFileChange} />
                      </label>
                  </div>

                  <button
                      type="submit"
                      disabled={loading}
                     className="flex flex-col items-center justify-center w-1/3 h-8 border-2 border-gray-300 rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 mt-4"
                  >
                      {loading ? 'Uploading..' : 'Upload'}
                  </button>
              </form>
          </div></>
  );
};

export default Upload;

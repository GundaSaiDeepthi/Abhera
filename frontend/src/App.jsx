import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Chatbot from './pages/Chatbot';
import Processing from './pages/Processing';
import Report from './pages/Report';

function App() {
  return (
    <Router>
      <div className="app-layout">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/chatbot" element={<Chatbot />} />
            <Route path="/processing" element={<Processing />} />
            <Route path="/report" element={<Report />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;

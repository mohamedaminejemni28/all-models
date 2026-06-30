import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Home from './pages/Home';
import ModelsOverview from './pages/ModelsOverview';
import ModelDetail from './pages/ModelDetail';
import Comparison from './pages/Comparison';
import Datasets from './pages/Datasets';
import Biomechanics from './pages/Biomechanics';
import GaitResearchAssistant from './components/GaitResearchAssistant';

export default function App() {
  return (
    <Router>
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/models" element={<ModelsOverview />} />
            <Route path="/models/:modelName" element={<ModelDetail />} />
            <Route path="/comparison" element={<Comparison />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/biomechanics" element={<Biomechanics />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <GaitResearchAssistant />
      </div>
    </Router>
  );
}

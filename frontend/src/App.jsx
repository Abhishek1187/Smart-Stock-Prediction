import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import TerminalDashboard from "./pages/TerminalDashboard.jsx";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<TerminalDashboard />} />
        <Route path="/terminal" element={<TerminalDashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
};

export default App;

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout/Layout';
import { Playground } from './pages/Playground';
import { NewTraining } from './pages/Train/NewTraining';
import { Monitor } from './pages/Train/Monitor';
import { Experiments } from './pages/Train/Experiments';
import { Models } from './pages/Train/Models';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Playground />} />
          <Route path="train">
            <Route path="new" element={<NewTraining />} />
            <Route path="monitor" element={<Monitor />} />
            <Route path="experiments" element={<Experiments />} />
            <Route path="models" element={<Models />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;

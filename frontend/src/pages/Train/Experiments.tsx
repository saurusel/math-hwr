import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { listExperiments } from '../../api/training';
import { duplicateRun } from '../../api/experiments';
import { ExperimentsComparison } from '../../components/Train/ExperimentsComparison';
import { useToast } from '../../components/Toast/useToast';
import { Toast } from '../../components/Toast/Toast';
import type { Experiment } from '../../types/training';

export function Experiments() {
  const navigate = useNavigate();
  const { toasts, removeToast, success, error } = useToast();
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [filteredExperiments, setFilteredExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [modelFilter, setModelFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  // Selection & Comparison
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showComparison, setShowComparison] = useState(false);

  useEffect(() => {
    listExperiments()
      .then((exps) => {
        setExperiments(exps);
        setFilteredExperiments(exps);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  // Apply filters
  useEffect(() => {
    let filtered = experiments;

    if (searchQuery) {
      filtered = filtered.filter((exp) =>
        exp.run_name.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    if (modelFilter !== 'all') {
      filtered = filtered.filter((exp) => exp.model_type === modelFilter);
    }

    if (statusFilter !== 'all') {
      filtered = filtered.filter((exp) => exp.status === statusFilter);
    }

    setFilteredExperiments(filtered);
  }, [experiments, searchQuery, modelFilter, statusFilter]);

  const handleSelectToggle = (expId: string) => {
    const newSelected = new Set(selectedIds);
    if (newSelected.has(expId)) {
      newSelected.delete(expId);
    } else {
      newSelected.add(expId);
    }
    setSelectedIds(newSelected);
  };

  const handleSelectAll = () => {
    if (selectedIds.size === filteredExperiments.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filteredExperiments.map((e) => e.run_id)));
    }
  };

  const handleCompare = () => {
    if (selectedIds.size < 2) {
      alert('Please select at least 2 experiments to compare');
      return;
    }
    setShowComparison(true);
  };

  const handleDuplicate = async (runId: string) => {
    try {
      const config = await duplicateRun(runId);
      // Navigate to new training page with config
      navigate('/train/new', { state: { config } });
      success('Configuration duplicated. Modify and start training.');
    } catch (err) {
      error(err instanceof Error ? err.message : 'Failed to duplicate run');
    }
  };

  const selectedExperiments = experiments.filter((e) => selectedIds.has(e.run_id));

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold text-slate-800">Эксперименты</h1>
        <div className="flex gap-3">
          {selectedIds.size > 0 && (
            <button
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              onClick={handleCompare}
            >
              📊 Сравнить ({selectedIds.size})
            </button>
          )}
          <Link
            to="/train/new"
            className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700"
          >
            + Новое обучение
          </Link>
        </div>
      </div>

      {/* Filters */}
      {!loading && experiments.length > 0 && (
        <div className="bg-white rounded-xl shadow p-4 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="md:col-span-2">
              <input
                type="text"
                placeholder="🔍 Поиск по названию..."
                className="w-full px-4 py-2 border border-slate-300 rounded-lg"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <div>
              <select
                className="w-full px-4 py-2 border border-slate-300 rounded-lg"
                value={modelFilter}
                onChange={(e) => setModelFilter(e.target.value)}
              >
                <option value="all">Все модели</option>
                <option value="M1">M1 (CRNN-CTC)</option>
                <option value="M2">M2 (Сегментация + MLP)</option>
                <option value="M3">M3 (Классификатор символов)</option>
              </select>
            </div>
            <div>
              <select
                className="w-full px-4 py-2 border border-slate-300 rounded-lg"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <option value="all">Все статусы</option>
                <option value="RUNNING">Обучается</option>
                <option value="FINISHED">Завершено</option>
                <option value="FAILED">Ошибка</option>
                <option value="STOPPED">Остановлено</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <div className="bg-white rounded-xl shadow p-12 text-center text-slate-400">
          Загрузка экспериментов...
        </div>
      ) : filteredExperiments.length === 0 ? (
        <div className="bg-white rounded-xl shadow p-12 text-center">
          {experiments.length === 0 ? (
            <>
              <p className="text-slate-600 mb-4">Экспериментов пока нет</p>
              <Link
                to="/train/new"
                className="inline-block px-6 py-3 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700"
              >
                Создать первое обучение
              </Link>
            </>
          ) : (
            <p className="text-slate-600">Нет экспериментов, соответствующих фильтрам</p>
          )}
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow overflow-hidden">
          <table className="w-full">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="px-4 py-3 text-left">
                  <input
                    type="checkbox"
                    checked={selectedIds.size === filteredExperiments.length}
                    onChange={handleSelectAll}
                    className="rounded"
                  />
                </th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-slate-700">
                  Название
                </th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-slate-700">
                  Модель
                </th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-slate-700">
                  Статус
                </th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-slate-700">
                  Начало
                </th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-slate-700">
                  Действия
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {filteredExperiments.map((exp) => (
                <tr key={exp.run_id} className="hover:bg-slate-50">
                  <td className="px-4 py-4">
                    <input
                      type="checkbox"
                      checked={selectedIds.has(exp.run_id)}
                      onChange={() => handleSelectToggle(exp.run_id)}
                      className="rounded"
                    />
                  </td>
                  <td className="px-6 py-4 text-sm font-medium text-slate-800">
                    {exp.run_name}
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-600">{exp.model_type}</td>
                  <td className="px-6 py-4">
                    <span
                      className={`px-2 py-1 rounded text-xs ${
                        exp.status === 'RUNNING'
                          ? 'bg-emerald-100 text-emerald-700'
                          : exp.status === 'FINISHED'
                          ? 'bg-blue-100 text-blue-700'
                          : exp.status === 'FAILED'
                          ? 'bg-red-100 text-red-700'
                          : 'bg-slate-100 text-slate-700'
                      }`}
                    >
                      {exp.status === 'RUNNING' ? 'Обучается' :
                       exp.status === 'FINISHED' ? 'Завершено' :
                       exp.status === 'FAILED' ? 'Ошибка' :
                       exp.status === 'STOPPED' ? 'Остановлено' : exp.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-600">
                    {new Date(exp.started_at).toLocaleString('ru-RU')}
                  </td>
                  <td className="px-6 py-4 text-sm">
                    <div className="flex items-center gap-2">
                      <Link
                        to={`/train/monitor?job=${exp.run_id}`}
                        className="px-3 py-1 bg-emerald-600 text-white rounded hover:bg-emerald-700 text-xs font-medium"
                      >
                        📊 Монитор
                      </Link>
                      <button
                        onClick={async () => {
                          if (confirm(`Удалить обучение "${exp.run_name}"?\n\nВсе чекпоинты и данные будут удалены без возможности восстановления.`)) {
                            try {
                              // TODO: implement deleteRun API
                              alert('Функция удаления будет реализована');
                            } catch (err) {
                              alert('Ошибка удаления');
                            }
                          }
                        }}
                        className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-xs font-medium"
                        title="Удалить обучение"
                      >
                        🗑️ Удалить
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Comparison Modal */}
      {showComparison && (
        <ExperimentsComparison
          experiments={selectedExperiments}
          onClose={() => setShowComparison(false)}
        />
      )}

      {/* Toast notifications */}
      <div className="fixed top-4 right-4 space-y-2 z-50">
        {toasts.map((toast) => (
          <Toast
            key={toast.id}
            message={toast.message}
            type={toast.type}
            onClose={() => removeToast(toast.id)}
          />
        ))}
      </div>
    </div>
  );
}

import { useState } from 'react';
import { addRunTag, removeRunTag, updateRunNotes } from '../../api/experiments';
import { useToast } from '../Toast/useToast';

interface NotesAndTagsProps {
  runId: string;
  initialNotes?: string;
  initialTags?: string[];
}

export function NotesAndTags({ runId, initialNotes = '', initialTags = [] }: NotesAndTagsProps) {
  const [notes, setNotes] = useState(initialNotes);
  const [tags, setTags] = useState<string[]>(initialTags);
  const [newTag, setNewTag] = useState('');
  const [isSavingNotes, setIsSavingNotes] = useState(false);

  const { success, error } = useToast();

  const handleSaveNotes = async () => {
    setIsSavingNotes(true);
    try {
      await updateRunNotes(runId, notes);
      success('Notes saved successfully');
    } catch (err) {
      error(err instanceof Error ? err.message : 'Failed to save notes');
    } finally {
      setIsSavingNotes(false);
    }
  };

  const handleAddTag = async () => {
    const trimmed = newTag.trim();
    if (!trimmed || tags.includes(trimmed)) {
      setNewTag('');
      return;
    }

    try {
      await addRunTag(runId, trimmed);
      setTags([...tags, trimmed]);
      setNewTag('');
      success(`Tag "${trimmed}" added`);
    } catch (err) {
      error(err instanceof Error ? err.message : 'Failed to add tag');
    }
  };

  const handleRemoveTag = async (tag: string) => {
    try {
      await removeRunTag(runId, tag);
      setTags(tags.filter((t) => t !== tag));
      success(`Tag "${tag}" removed`);
    } catch (err) {
      error(err instanceof Error ? err.message : 'Failed to remove tag');
    }
  };

  return (
    <div className="space-y-4">
      {/* Tags */}
      <div>
        <label className="block text-sm font-medium text-slate-700 mb-2">Tags</label>
        <div className="flex flex-wrap gap-2 mb-2">
          {tags.map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center gap-1 px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
            >
              {tag}
              <button
                onClick={() => handleRemoveTag(tag)}
                className="hover:text-blue-900"
                title="Remove tag"
              >
                ×
              </button>
            </span>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Add tag..."
            className="flex-1 px-3 py-2 border border-slate-300 rounded-lg text-sm"
            value={newTag}
            onChange={(e) => setNewTag(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAddTag()}
          />
          <button
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
            onClick={handleAddTag}
          >
            + Add
          </button>
        </div>
      </div>

      {/* Notes */}
      <div>
        <label className="block text-sm font-medium text-slate-700 mb-2">Notes</label>
        <textarea
          className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm"
          rows={4}
          placeholder="Add notes about this experiment..."
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
        <button
          className={`mt-2 px-4 py-2 rounded-lg text-sm ${
            isSavingNotes
              ? 'bg-slate-400 cursor-not-allowed'
              : 'bg-emerald-600 hover:bg-emerald-700 text-white'
          }`}
          onClick={handleSaveNotes}
          disabled={isSavingNotes}
        >
          {isSavingNotes ? 'Saving...' : 'Save Notes'}
        </button>
      </div>
    </div>
  );
}

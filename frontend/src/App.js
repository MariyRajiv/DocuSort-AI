import { useState, useCallback, useRef } from "react";
import "@/App.css";
import axios from "axios";
import { Upload, FileText, CheckCircle, AlertCircle, Download, History, Eye, X, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [selectedText, setSelectedText] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState([]);
  const [selectedHistoryItem, setSelectedHistoryItem] = useState(null);
  const fileInputRef = useRef(null);

  const loadHistory = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/history`);
      setHistory(response.data);
    } catch (error) {
      console.error("Error loading history:", error);
      toast.error("Failed to load history");
    }
  }, []);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  };

  const handleFileInput = (e) => {
    const selectedFiles = Array.from(e.target.files);
    handleFiles(selectedFiles);
  };

  const handleFiles = (newFiles) => {
    // Check file sizes
    const validFiles = newFiles.filter(file => {
      if (file.size > 50 * 1024 * 1024) {
        toast.error(`${file.name} exceeds 50MB limit`);
        return false;
      }
      return true;
    });

    setFiles(prev => [...prev, ...validFiles]);
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const processFiles = async () => {
    if (files.length === 0) {
      toast.error("Please select files to classify");
      return;
    }

    setProcessing(true);
    setProgress(0);
    setResults([]);

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });

      const response = await axios.post(`${API}/classify`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
        }
      });

      setResults(response.data);
      toast.success(`Successfully classified ${response.data.length} document(s)`);
      setFiles([]);
      
      // Reload history
      await loadHistory();
    } catch (error) {
      console.error("Classification error:", error);
      toast.error("Failed to classify documents");
    } finally {
      setProcessing(false);
      setProgress(0);
    }
  };

  const downloadResults = () => {
    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `classification-results-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success("Results downloaded");
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return "bg-emerald-50 text-emerald-700 border-emerald-200";
    if (confidence >= 0.6) return "bg-blue-50 text-blue-700 border-blue-200";
    if (confidence >= 0.4) return "bg-amber-50 text-amber-700 border-amber-200";
    return "bg-rose-50 text-rose-700 border-rose-200";
  };

  const getDocTypeColor = (docType) => {
    if (docType.includes("Resume")) return "bg-purple-50 text-purple-700 border-purple-200";
    if (docType.includes("Invoice")) return "bg-cyan-50 text-cyan-700 border-cyan-200";
    if (docType.includes("Contract")) return "bg-indigo-50 text-indigo-700 border-indigo-200";
    if (docType.includes("Report")) return "bg-orange-50 text-orange-700 border-orange-200";
    return "bg-slate-50 text-slate-700 border-slate-200";
  };

  const loadHistoryItemDetails = async (itemId) => {
    try {
      // Get full details from MongoDB
      const response = await axios.get(`${API}/history`);
      const fullHistory = response.data;
      const item = fullHistory.find(h => h.id === itemId);
      if (item) {
        setSelectedHistoryItem(item);
      }
    } catch (error) {
      console.error("Error loading history item:", error);
      toast.error("Failed to load document details");
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <Toaster position="top-right" />
      
      {/* Header */}
      <header className="border-b border-zinc-200 bg-gradient-to-r from-white via-blue-50 to-white shadow-sm">
        <div className="max-w-7xl mx-auto px-8 py-6 flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-extrabold tracking-tighter bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent" style={{ fontFamily: 'Manrope' }}>
              DocuSort AI
            </h1>
            <p className="text-sm text-zinc-600 mt-1 font-medium" style={{ fontFamily: 'Public Sans' }}>
              Universal Document Classification System
            </p>
          </div>
          <Button
            variant="outline"
            onClick={() => {
              setShowHistory(true);
              loadHistory();
            }}
            data-testid="history-button"
            className="flex items-center gap-2 border-blue-300 text-blue-600 hover:bg-blue-50 hover:border-blue-400 font-medium shadow-sm"
          >
            <History className="h-4 w-4" />
            History
          </Button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-8 py-12">
        {/* Upload Zone */}
        {results.length === 0 && (
          <div className="mb-12">
            <div
              data-testid="upload-zone"
              className={`border-2 border-dashed rounded-xl p-20 flex flex-col items-center justify-center text-center transition-all cursor-pointer group ${
                isDragging 
                  ? 'border-blue-500 bg-gradient-to-br from-blue-50 via-cyan-50 to-purple-50 ring-4 ring-blue-500/20 shadow-lg' 
                  : 'border-zinc-300 bg-gradient-to-br from-white via-zinc-50 to-blue-50/30 hover:border-blue-400 hover:shadow-md'
              }`}
              onDragEnter={handleDragEnter}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className={`p-5 rounded-full mb-6 transition-all ${
                isDragging ? 'bg-blue-500 shadow-lg' : 'bg-gradient-to-br from-blue-100 to-purple-100 group-hover:from-blue-200 group-hover:to-purple-200'
              }`}>
                <Upload className={`h-16 w-16 transition-colors ${
                  isDragging ? 'text-white' : 'text-blue-600 group-hover:text-blue-700'
                }`} />
              </div>
              <h3 className="text-3xl font-bold text-zinc-900 mb-3 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent" style={{ fontFamily: 'Manrope' }}>
                Drop your documents here
              </h3>
              <p className="text-zinc-600 mb-6 font-medium" style={{ fontFamily: 'Public Sans' }}>
                or click to browse files (PDF, DOCX, Images, etc.)
              </p>
              <div className="flex items-center gap-4 text-sm text-zinc-600 bg-white/80 px-6 py-3 rounded-full border border-zinc-200">
                <span className="font-mono font-semibold text-blue-600">Max 50MB per file</span>
                <span className="text-zinc-400">•</span>
                <span className="font-mono font-semibold text-purple-600">Unlimited files</span>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                onChange={handleFileInput}
                className="hidden"
                accept=".pdf,.docx,.doc,.txt,.jpg,.jpeg,.png,.pptx,.ppt"
              />
            </div>

            {/* Selected Files */}
            {files.length > 0 && (
              <div className="mt-8">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-xl font-bold text-zinc-900" style={{ fontFamily: 'Manrope' }}>
                      Selected Files
                    </h3>
                    <p className="text-sm text-zinc-600 mt-1">
                      <span className="font-mono text-blue-600 font-semibold">{files.length}</span> file(s) ready to classify
                    </p>
                  </div>
                  <Button
                    onClick={processFiles}
                    disabled={processing}
                    data-testid="classify-button"
                    className="bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700 rounded-lg px-8 py-3 font-bold shadow-lg hover:shadow-xl transition-all active:scale-95"
                  >
                    {processing ? (
                      <>
                        <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <CheckCircle className="h-5 w-5 mr-2" />
                        Classify Documents
                      </>
                    )}
                  </Button>
                </div>

                {processing && (
                  <div className="mb-6 bg-gradient-to-r from-blue-50 to-purple-50 p-5 rounded-xl border border-blue-200">
                    <Progress value={progress} className="h-3 mb-3" />
                    <p className="text-sm text-blue-700 font-bold text-center" style={{ fontFamily: 'JetBrains Mono' }}>
                      {progress}% complete - Processing documents...
                    </p>
                  </div>
                )}

                <div className="space-y-3">
                  {files.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-5 border-2 border-zinc-200 rounded-xl bg-gradient-to-r from-white to-zinc-50 hover:border-blue-300 hover:shadow-md transition-all"
                    >
                      <div className="flex items-center gap-4">
                        <div className="p-3 bg-gradient-to-br from-blue-100 to-purple-100 rounded-lg">
                          <FileText className="h-6 w-6 text-blue-600" />
                        </div>
                        <div>
                          <p className="text-base font-semibold text-zinc-900">{file.name}</p>
                          <p className="text-sm text-zinc-500 font-mono mt-1" style={{ fontFamily: 'JetBrains Mono' }}>
                            {formatFileSize(file.size)}
                          </p>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeFile(index);
                        }}
                        data-testid={`remove-file-${index}`}
                        className="hover:bg-red-50 hover:text-red-600 border border-transparent hover:border-red-200"
                      >
                        <X className="h-5 w-5" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-8 bg-gradient-to-r from-emerald-50 via-blue-50 to-purple-50 p-6 rounded-2xl border border-emerald-200 shadow-sm">
              <div>
                <h2 className="text-3xl font-bold bg-gradient-to-r from-emerald-600 to-blue-600 bg-clip-text text-transparent" style={{ fontFamily: 'Manrope' }}>
                  Classification Results
                </h2>
                <p className="text-zinc-700 mt-2 font-medium" style={{ fontFamily: 'Public Sans' }}>
                  <CheckCircle className="h-5 w-5 inline text-emerald-600 mr-2" />
                  Successfully classified <span className="font-bold text-emerald-600 font-mono">{results.length}</span> document(s)
                </p>
              </div>
              <div className="flex gap-3">
                <Button
                  variant="outline"
                  onClick={downloadResults}
                  data-testid="download-results-button"
                  className="flex items-center gap-2 border-blue-300 text-blue-600 hover:bg-blue-50 hover:border-blue-400 font-medium shadow-sm"
                >
                  <Download className="h-4 w-4" />
                  Download JSON
                </Button>
                <Button
                  onClick={() => {
                    setResults([]);
                    setFiles([]);
                  }}
                  data-testid="classify-more-button"
                  className="bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700 shadow-lg hover:shadow-xl font-bold"
                >
                  Classify More
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {results.map((result, index) => (
                <Card
                  key={result.id}
                  data-testid={`result-card-${index}`}
                  className="border border-zinc-200 shadow-lg hover:shadow-xl hover:border-blue-300 transition-all bg-gradient-to-br from-white to-zinc-50"
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between mb-3">
                      <Badge className="bg-zinc-900 text-white font-mono text-xs px-3 py-1">{result.file_type}</Badge>
                      <Badge
                        className={`${getConfidenceColor(result.confidence)} font-mono text-xs px-3 py-1 font-bold`}
                        style={{ fontFamily: 'JetBrains Mono' }}
                      >
                        {(result.confidence * 100).toFixed(0)}%
                      </Badge>
                    </div>
                    <CardTitle className="text-lg font-bold text-zinc-900 break-words leading-tight" style={{ fontFamily: 'Manrope' }}>
                      {result.file_name}
                    </CardTitle>
                    <CardDescription className="text-sm text-zinc-500 font-mono mt-1" style={{ fontFamily: 'Public Sans' }}>
                      {formatFileSize(result.file_size)}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-3 rounded-lg border border-purple-200">
                        <p className="text-xs uppercase tracking-widest text-purple-700 mb-1 font-bold">
                          DOCUMENT TYPE
                        </p>
                        <Badge className={`${getDocTypeColor(result.document_type)} text-sm font-medium mt-1`}>
                          {result.document_type}
                        </Badge>
                      </div>
                      
                      <div className="bg-gradient-to-r from-blue-50 to-cyan-50 p-3 rounded-lg border border-blue-200">
                        <p className="text-xs uppercase tracking-widest text-blue-700 mb-1 font-bold">
                          SUMMARY
                        </p>
                        <p className="text-sm text-zinc-700 leading-relaxed mt-1">{result.summary}</p>
                      </div>

                      <div className="bg-gradient-to-r from-amber-50 to-orange-50 p-3 rounded-lg border border-amber-200">
                        <p className="text-xs uppercase tracking-widest text-amber-700 mb-1 font-bold">
                          REASON
                        </p>
                        <p className="text-xs text-zinc-600 leading-relaxed mt-1">{result.reason}</p>
                      </div>

                      {result.extracted_text && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedText(result)}
                          data-testid={`view-text-button-${index}`}
                          className="w-full mt-2 flex items-center justify-center gap-2 border-blue-300 text-blue-600 hover:bg-blue-50 hover:border-blue-400 font-medium"
                        >
                          <Eye className="h-4 w-4" />
                          View Extracted Text
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Text Viewer Dialog */}
      <Dialog open={!!selectedText} onOpenChange={() => setSelectedText(null)}>
        <DialogContent className="max-w-4xl max-h-[85vh]" data-testid="text-viewer-dialog">
          <DialogHeader>
            <DialogTitle className="text-2xl font-bold text-zinc-900 mb-2" style={{ fontFamily: 'Manrope' }}>
              {selectedText?.file_name}
            </DialogTitle>
            <DialogDescription className="text-zinc-600" style={{ fontFamily: 'Public Sans' }}>
              Extracted text content from document
            </DialogDescription>
          </DialogHeader>
          <div className="bg-gradient-to-r from-slate-50 to-zinc-50 p-1 rounded-xl">
            <ScrollArea className="h-[500px] w-full rounded-lg bg-white border-2 border-slate-200 p-6">
              <pre className="text-sm whitespace-pre-wrap font-mono text-zinc-700 leading-relaxed">
                {selectedText?.extracted_text || "No text extracted"}
              </pre>
            </ScrollArea>
          </div>
        </DialogContent>
      </Dialog>

      {/* History Dialog */}
      {/* History Dialog */}
      <Dialog open={showHistory} onOpenChange={setShowHistory}>
        <DialogContent className="max-w-5xl max-h-[85vh] flex flex-col overflow-hidden" data-testid="history-dialog">
          <DialogHeader>
            <DialogTitle className="text-2xl font-bold text-zinc-900" style={{ fontFamily: 'Manrope' }}>
              Classification History
            </DialogTitle>
            <DialogDescription className="text-zinc-600" style={{ fontFamily: 'Public Sans' }}>
              Recently classified documents - Click to view details
            </DialogDescription>
          </DialogHeader>
          <ScrollArea className="h-[550px] w-full">
            <div className="flex flex-col gap-4 pr-4">
              {history.length === 0 ? (
                <div className="text-center py-16">
                  <History className="h-16 w-16 text-zinc-300 mx-auto mb-4" />
                  <p className="text-zinc-500 font-medium">No history yet</p>
                  <p className="text-sm text-zinc-400 mt-1">Upload and classify documents to see them here</p>
                </div>
              ) : (
                history.map((item, index) => (
                  <div
                    key={item.id}
                    data-testid={`history-item-${index}`}
                    onClick={() => setSelectedHistoryItem(item)}
                    className="group cursor-pointer rounded-lg border border-zinc-200 p-4 flex items-center justify-between hover:border-blue-400 hover:shadow-lg bg-white transition"
                    style={{ userSelect: "none" }}
                  >
                    <div className="flex items-center gap-4 min-w-0 flex-1">
                      <div className={`p-3 rounded-lg ${getDocTypeColor(item.document_type)}`}>
                        <FileText className="h-6 w-6" />
                      </div>
                      <div className="flex flex-col min-w-0">
                        <p
                          className="text-base font-semibold text-zinc-900 truncate group-hover:text-blue-600 transition-colors"
                          style={{ fontFamily: "Manrope" }}
                        >
                          {item.file_name}
                        </p>
                        <div className="flex items-center gap-2 mt-1 text-xs text-zinc-600">
                          <Badge className={`${getDocTypeColor(item.document_type)} text-xs font-medium`}>
                            {item.document_type}
                          </Badge>
                          <span>•</span>
                          <span className="font-mono">{item.file_type}</span>
                          <span>•</span>
                          <span>{formatFileSize(item.file_size)}</span>
                        </div>
                        <p className="text-sm text-zinc-700 mt-2 line-clamp-2">{item.summary}</p>
                        {item.timestamp && (
                          <p className="text-xs text-zinc-400 mt-1">
                            {new Date(item.timestamp).toLocaleString()}
                          </p>
                        )}
                      </div>
                    </div>

                    <div className="flex flex-col items-end gap-2 ml-4">
                      <Badge
                        className={`${getConfidenceColor(item.confidence)} text-xs font-bold px-3 py-1`}
                        style={{ fontFamily: "JetBrains Mono" }}
                      >
                        {(item.confidence * 100).toFixed(0)}%
                      </Badge>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedHistoryItem(item);
                        }}
                        className="text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded px-2 py-1 flex items-center text-sm"
                      >
                        <Eye className="h-4 w-4 mr-1" />
                        View
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </ScrollArea>
        </DialogContent>
      </Dialog>


            {/* History Item Details Dialog */}
      <Dialog
        open={!!selectedHistoryItem}
        onOpenChange={(open) => {
          if (!open) setSelectedHistoryItem(null); // Close dialog
        }}
      >
        <DialogContent className="max-w-4xl w-full max-h-[85vh] flex flex-col" data-testid="history-details-dialog">
          <DialogHeader>
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1">
                <DialogTitle className="text-2xl font-bold text-zinc-900 mb-2" style={{ fontFamily: 'Manrope' }}>
                  {selectedHistoryItem?.file_name || "No file selected"}
                </DialogTitle>
                <div className="flex items-center gap-2 flex-wrap">
                  <Badge className={`${getDocTypeColor(selectedHistoryItem?.document_type || '')} text-sm font-medium`}>
                    {selectedHistoryItem?.document_type}
                  </Badge>
                  <Badge className="bg-zinc-100 text-zinc-700 text-sm font-mono">
                    {selectedHistoryItem?.file_type}
                  </Badge>
                  <Badge
                    className={`${getConfidenceColor(selectedHistoryItem?.confidence || 0)} text-sm font-bold`}
                    style={{ fontFamily: 'JetBrains Mono' }}
                  >
                    {((selectedHistoryItem?.confidence || 0) * 100).toFixed(0)}% Confidence
                  </Badge>
                </div>
              </div>
            </div>
          </DialogHeader>

          {/* Scrollable content */}
          <div className="flex-1 overflow-auto mt-4 pr-2">
            <div className="space-y-6">

              {/* File Info */}
              <div className="bg-gradient-to-r from-blue-50 to-cyan-50 p-5 rounded-xl border border-blue-200">
                <h3 className="text-sm font-bold uppercase tracking-wider text-blue-900 mb-3" style={{ fontFamily: 'Manrope' }}>
                  File Information
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-blue-600 font-semibold mb-1">File Size</p>
                    <p className="text-sm text-zinc-900 font-mono">{formatFileSize(selectedHistoryItem?.file_size || 0)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-blue-600 font-semibold mb-1">Classified On</p>
                    <p className="text-sm text-zinc-900">
                      {selectedHistoryItem?.timestamp
                        ? new Date(selectedHistoryItem.timestamp).toLocaleString()
                        : 'N/A'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Summary */}
              <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-5 rounded-xl border border-purple-200">
                <h3 className="text-sm font-bold uppercase tracking-wider text-purple-900 mb-2" style={{ fontFamily: 'Manrope' }}>
                  Summary
                </h3>
                <p className="text-sm text-zinc-700 leading-relaxed">{selectedHistoryItem?.summary}</p>
              </div>

              {/* Reason */}
              <div className="bg-gradient-to-r from-amber-50 to-orange-50 p-5 rounded-xl border border-amber-200">
                <h3 className="text-sm font-bold uppercase tracking-wider text-amber-900 mb-2" style={{ fontFamily: 'Manrope' }}>
                  Classification Reason
                </h3>
                <p className="text-sm text-zinc-700 leading-relaxed">{selectedHistoryItem?.reason}</p>
              </div>

              {/* Extracted Text */}
              {selectedHistoryItem?.extracted_text && (
                <div className="bg-gradient-to-r from-slate-50 to-zinc-50 p-5 rounded-xl border border-slate-200">
                  <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900 mb-3" style={{ fontFamily: 'Manrope' }}>
                    Extracted Text
                  </h3>
                  <div className="bg-white rounded-lg border border-slate-200 p-4 max-h-[300px] overflow-auto">
                    <pre className="text-xs whitespace-pre-wrap font-mono text-zinc-700 leading-relaxed">
                      {selectedHistoryItem.extracted_text || "No text extracted"}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

    </div>
  );
}

export default App;

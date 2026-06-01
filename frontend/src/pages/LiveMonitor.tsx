import React, { useState, useEffect, useRef } from 'react';
import { Grid, Card, CardContent, Typography, Box, Select, MenuItem, InputLabel, FormControl, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip } from '@mui/material';
import { Play, Tv, ShieldAlert } from 'lucide-react';

export default function LiveMonitor() {
  const [cameras, setCameras] = useState<any[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>("cam_01");
  const [activeFrame, setActiveFrame] = useState<string>("");
  const [analytics, setAnalytics] = useState<any[]>([]);
  const [fps, setFps] = useState<number>(0.0);
  const [socketStatus, setSocketStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  
  const ws = useRef<WebSocket | null>(null);

  // Fetch available cameras on load
  useEffect(() => {
    fetch("http://localhost:8000/api/cameras")
      .then(res => res.json())
      .then(data => {
        setCameras(data);
        if (data.length > 0) {
          setSelectedCamera(data[0].id);
        }
      })
      .catch(() => {
        setCameras([{ id: "cam_01", name: "Default Webcam", url: "0" }]);
      });
  }, []);

  // Establish WebSockets feed connection matching the selected camera ID
  useEffect(() => {
    if (!selectedCamera) return;
    
    setSocketStatus('connecting');
    ws.current = new WebSocket(`ws://localhost:8000/api/ws/monitor/${selectedCamera}`);
    
    ws.current.onopen = () => {
      setSocketStatus('connected');
      console.log(`WebSocket channel successfully opened for ${selectedCamera}`);
    };
    
    ws.current.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        setActiveFrame(payload.frame);
        setAnalytics(payload.analytics);
        setFps(payload.fps);
      } catch (err) {
        console.error("Error decoding frames from channel stream", err);
      }
    };
    
    ws.current.onerror = () => {
      setSocketStatus('disconnected');
    };
    
    ws.current.onclose = () => {
      setSocketStatus('disconnected');
    };
    
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [selectedCamera]);

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Tv size={24} color="#00f2fe" />
          <Typography variant="body1" sx={{ color: 'text.secondary', fontWeight: 500 }}>Select camera to stream:</Typography>
        </Box>
        <FormControl variant="standard" sx={{ minWidth: 200 }}>
          <Select
            value={selectedCamera}
            onChange={(e) => setSelectedCamera(e.target.value)}
            sx={{ color: '#00f2fe', fontWeight: 600, borderBottom: '1px solid rgba(0,242,254,0.3)' }}
          >
            {cameras.map((cam) => (
              <MenuItem key={cam.id} value={cam.id}>{cam.name}</MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      <Grid container spacing={4}>
        {/* Stream Board Window */}
        <Grid item xs={12} lg={8}>
          <Card className="glass-panel" sx={{ bgcolor: 'transparent', overflow: 'hidden' }}>
            <Box sx={{ position: 'relative', minHeight: 480, display: 'flex', justifyContent: 'center', alignItems: 'center', bgcolor: '#04030a' }}>
              {activeFrame ? (
                <img src={activeFrame} alt="Live analytical feed" style={{ width: '100%', height: 'auto', display: 'block' }} />
              ) : (
                <Box sx={{ textAlign: 'center', p: 4 }}>
                  <Play size={48} color="#00f2fe" style={{ opacity: 0.5, marginBottom: 16 }} />
                  <Typography variant="h6" sx={{ color: 'text.secondary', fontWeight: 600 }}>
                    {socketStatus === 'connecting' ? 'CONNECTING TO CAMERA...' : 'CAMERA STANDBY FEED'}
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                    Websocket channel status: {socketStatus}
                  </Typography>
                </Box>
              )}
              
              {/* Overlay Indicators */}
              <Box sx={{ position: 'absolute', top: 16, right: 16, display: 'flex', gap: 1.5 }}>
                <Chip
                  label={`FPS: ${fps}`}
                  sx={{ bgcolor: 'rgba(15,14,31,0.85)', border: '1px solid rgba(0,242,254,0.3)', color: '#00f2fe', fontWeight: 700 }}
                />
                <Chip
                  label={socketStatus.toUpperCase()}
                  color={socketStatus === 'connected' ? 'success' : socketStatus === 'connecting' ? 'warning' : 'error'}
                  sx={{ fontWeight: 700 }}
                />
              </Box>
            </Box>
          </Card>
        </Grid>

        {/* Analytics Target Summary */}
        <Grid item xs={12} lg={4}>
          <Card className="glass-panel" sx={{ bgcolor: 'transparent', height: '100%', minHeight: 480 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 3, color: '#00f2fe', display: 'flex', alignItems: 'center', gap: 1 }}>
                <ShieldAlert size={20} /> Active Faces Telemetry
              </Typography>
              
              {analytics.length === 0 ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '350px' }}>
                  <Typography variant="body2" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
                    Scanning background for faces...
                  </Typography>
                </Box>
              ) : (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow sx={{ borderBottom: '1px solid rgba(0,242,254,0.1)' }}>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Face</TableCell>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Age & Gender</TableCell>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Emotion</TableCell>
                        <TableCell sx={{ color: 'text.secondary', fontWeight: 600 }}>Liveness</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {analytics.map((face, index) => (
                        <TableRow key={index} sx={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <TableCell sx={{ fontWeight: 700, color: face.name === 'Unknown' ? '#ff0844' : '#00f2fe' }}>
                            {face.name}
                          </TableCell>
                          <TableCell>{face.gender}, {face.age}</TableCell>
                          <TableCell>{face.emotion}</TableCell>
                          <TableCell>
                            <Chip
                              label={face.is_spoof ? 'SPOOF' : 'LIVE'}
                              color={face.is_spoof ? 'error' : 'success'}
                              size="small"
                              sx={{ fontWeight: 800, fontSize: '10px' }}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

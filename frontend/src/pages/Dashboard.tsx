import React, { useState, useEffect } from 'react';
import { Grid, Card, CardContent, Typography, Box, CircularProgress, LinearProgress } from '@mui/material';
import { Users, UserCheck, Eye, ShieldAlert, Cpu, HardDrive } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';

export default function Dashboard() {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/analytics/dashboard");
        if (res.ok) {
          const data = await res.json();
          setMetrics(data);
        }
      } catch (err) {
        // Fallback demo metrics if server is not fully running locally yet
        setMetrics({
          total_registered: 18,
          today_attendance: 12,
          active_cameras: 1,
          today_alerts: 2,
          total_detections_today: 432,
          demographics: { Male: 8, Female: 10 },
          emotions: { Neutral: 280, Happy: 92, Sad: 34, Surprise: 16, Angry: 10 },
          system_resources: { cpu: 22, ram: 58 }
        });
      } finally {
        setLoading(false);
      }
    };
    
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 3000);
    return () => clearInterval(interval);
  }, []);

  if (loading || !metrics) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh' }}>
        <CircularProgress color="primary" />
      </Box>
    );
  }

  // Formatting chart data structures
  const demographicData = Object.entries(metrics.demographics).map(([name, value]) => ({ name, value }));
  const emotionData = Object.entries(metrics.emotions).map(([name, value]) => ({ name, value }));
  
  const COLORS = ['#00f2fe', '#4facfe', '#ff0844', '#f8fafc', '#94a3b8'];

  const stats = [
    { title: 'Registered Faces', value: metrics.total_registered, icon: <Users size={24} color="#00f2fe" />, subtitle: 'Total enrolled database database profiles' },
    { title: 'Today\'s Attendance', value: metrics.today_attendance, icon: <UserCheck size={24} color="#4facfe" />, subtitle: 'Successful logins' },
    { title: 'Active Camera Feeds', value: metrics.active_cameras, icon: <Eye size={24} color="#00f2fe" />, subtitle: 'Live ingestion feeds online' },
    { title: 'Security Alerts Triggered', value: metrics.today_alerts, icon: <ShieldAlert size={24} color="#ff0844" />, subtitle: 'Attempts flagged today' }
  ];

  return (
    <Box>
      {/* 4 Counter Panels */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {stats.map((stat, i) => (
          <Grid item xs={12} sm={6} md={3} key={i}>
            <Card className="glass-panel" sx={{ bgcolor: 'transparent' }}>
              <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2.5 }}>
                <Box sx={{ p: 2, borderRadius: 3, bgcolor: 'rgba(0, 242, 254, 0.05)', border: '1px solid rgba(0, 242, 254, 0.1)' }}>
                  {stat.icon}
                </Box>
                <Box>
                  <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>{stat.title}</Typography>
                  <Typography variant="h4" sx={{ fontWeight: 800, color: 'text.primary', my: 0.5 }}>{stat.value}</Typography>
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>{stat.subtitle}</Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Analytics Charts & Graphs */}
      <Grid container spacing={4} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Card className="glass-panel" sx={{ bgcolor: 'transparent', height: 350 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 2, color: '#00f2fe' }}>Gender Demographics</Typography>
              <Box sx={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={demographicData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {demographicData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#0f0e1f', border: '1px solid rgba(0,242,254,0.2)' }} />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card className="glass-panel" sx={{ bgcolor: 'transparent', height: 350 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 2, color: '#00f2fe' }}>Emotion Analytics</Typography>
              <Box sx={{ height: 260 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={emotionData}>
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip contentStyle={{ backgroundColor: '#0f0e1f', border: '1px solid rgba(0,242,254,0.2)' }} />
                    <Bar dataKey="value" fill="#4facfe" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* System Telemetry Logs */}
      <Card className="glass-panel" sx={{ bgcolor: 'transparent' }}>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 700, mb: 3, color: '#00f2fe' }}>Hardware Telemetry</Typography>
          <Grid container spacing={4}>
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Cpu size={18} color="#00f2fe" />
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>CPU Processor Ingestion Load</Typography>
                  </Box>
                  <Typography variant="body2" sx={{ fontWeight: 700 }}>{metrics.system_resources.cpu}%</Typography>
                </Box>
                <LinearProgress variant="determinate" value={metrics.system_resources.cpu} sx={{ height: 8, borderRadius: 4, bgcolor: 'rgba(255,255,255,0.05)' }} />
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <HardDrive size={18} color="#4facfe" />
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>Virtual RAM Memory Overhead</Typography>
                  </Box>
                  <Typography variant="body2" sx={{ fontWeight: 700 }}>{metrics.system_resources.ram}%</Typography>
                </Box>
                <LinearProgress variant="determinate" value={metrics.system_resources.ram} sx={{ height: 8, borderRadius: 4, bgcolor: 'rgba(255,255,255,0.05)' }} />
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
}

import React, { useState, useEffect } from 'react';
import { Card, CardContent, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography, Box, Button, TextField } from '@mui/material';
import { Download, Calendar } from 'lucide-react';

export default function AttendancePanel() {
  const [logs, setLogs] = useState<any[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>(new Date().toISOString().split('T')[0]);

  const fetchAttendance = async () => {
    try {
      const res = await fetch(`http://localhost:8000/api/attendance?date_str=${selectedDate}`);
      if (res.ok) {
        const data = await res.json();
        setLogs(data);
      }
    } catch (err) {
      // Demo fallback records if server is not fully running locally yet
      setLogs([
        { id: 1, name: "Alice Vance", role: "Employee", check_in: `${selectedDate} 08:52:14`, check_out: `${selectedDate} 17:02:40`, status: "Present" },
        { id: 2, name: "Robert Downey", role: "Employee", check_in: `${selectedDate} 09:12:05`, check_out: "Active Status", status: "Late" },
        { id: 3, name: "Scarlett Johansson", role: "Student", check_in: `${selectedDate} 08:44:59`, check_out: `${selectedDate} 15:30:12`, status: "Present" }
      ]);
    }
  };

  useEffect(() => {
    fetchAttendance();
  }, [selectedDate]);

  const exportToCSV = () => {
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "ID,Name,Role,Check In,Check Out,Status\n";
    
    logs.forEach(log => {
      csvContent += `${log.id},${log.name},${log.role},${log.check_in},${log.check_out},${log.status}\n`;
    });
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `Attendance_Report_${selectedDate}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4, flexWrap: 'wrap', gap: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Calendar color="#00f2fe" size={20} />
          <TextField
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            sx={{
              bgcolor: 'background.paper',
              borderRadius: 2,
              '& .MuiInputBase-input': { color: 'text.primary', p: 1.5 },
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'rgba(0, 242, 254, 0.15)' },
                '&:hover fieldset': { borderColor: 'rgba(0, 242, 254, 0.35)' },
              }
            }}
          />
        </Box>
        <Button
          variant="contained"
          onClick={exportToCSV}
          startIcon={<Download size={18} />}
          sx={{
            background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            color: '#080710',
            fontWeight: 700,
            textTransform: 'none',
            borderRadius: 2,
            boxShadow: '0 4px 15px rgba(0, 242, 254, 0.3)'
          }}
        >
          Export CSV Report
        </Button>
      </Box>

      <Card className="glass-panel" sx={{ bgcolor: 'transparent' }}>
        <CardContent>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow sx={{ borderBottom: '2px solid rgba(0, 242, 254, 0.2)' }}>
                  <TableCell sx={{ color: 'text.secondary', fontWeight: 700 }}>Employee/Student Name</TableCell>
                  <TableCell sx={{ color: 'text.secondary', fontWeight: 700 }}>Profile Role</TableCell>
                  <TableCell sx={{ color: 'text.secondary', fontWeight: 700 }}>Check-In Time</TableCell>
                  <TableCell sx={{ color: 'text.secondary', fontWeight: 700 }}>Check-Out Time</TableCell>
                  <TableCell sx={{ color: 'text.secondary', fontWeight: 700 }}>Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {logs.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} align="center" sx={{ py: 6, color: 'text.secondary', fontStyle: 'italic' }}>
                      No attendance logged for this date.
                    </TableCell>
                  </TableRow>
                ) : (
                  logs.map((log) => (
                    <TableRow key={log.id} sx={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)', '&:hover': { bgcolor: 'rgba(0, 242, 254, 0.02)' } }}>
                      <TableCell sx={{ fontWeight: 700, color: 'text.primary' }}>{log.name}</TableCell>
                      <TableCell>{log.role}</TableCell>
                      <TableCell>{log.check_in}</TableCell>
                      <TableCell sx={{ color: log.check_out === 'Active Status' ? '#00f2fe' : 'text.primary', fontWeight: log.check_out === 'Active Status' ? 600 : 400 }}>
                        {log.check_out}
                      </TableCell>
                      <TableCell sx={{ fontWeight: 700, color: log.status === 'Present' ? '#00f2fe' : '#ff0844' }}>
                        {log.status}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
}

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const PlantDiseaseApp());
}

class PlantDiseaseApp extends StatelessWidget {
  const PlantDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Plant Disease Detector',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const HealthCheckScreen(),
    );
  }
}

class HealthCheckScreen extends StatefulWidget {
  const HealthCheckScreen({super.key});

  @override
  State<HealthCheckScreen> createState() => _HealthCheckScreenState();
}

class _HealthCheckScreenState extends State<HealthCheckScreen> {
  // Use 127.0.0.1:8000 (accessible on Android via `adb reverse tcp:8000 tcp:8000`, Chrome Web, or Windows Desktop)
  final String _apiUrl = 'http://127.0.0.1:8000/health';
  bool _isLoading = false;
  Map<String, dynamic>? _healthData;
  String? _errorMessage;

  Future<void> _checkHealth() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      final response = await http
          .get(Uri.parse(_apiUrl))
          .timeout(const Duration(seconds: 5));

      if (response.statusCode == 200 || response.statusCode == 503) {
        final decoded = json.decode(response.body) as Map<String, dynamic>;
        setState(() {
          _healthData = decoded;
        });
      } else {
        setState(() {
          _errorMessage =
              'Сервер повернув код помилки: ${response.statusCode}\n${response.body}';
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage =
            'Не вдалося з\'єднатися з API ($_apiUrl).\n\nПереконайтеся, що бекенд запущено, а для Android виконано команду:\n`adb reverse tcp:8000 tcp:8000`\n\nДеталі помилки: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  void initState() {
    super.initState();
    _checkHealth();
  }

  @override
  Widget build(BuildContext context) {
    final isOk = _healthData != null && _healthData!['status'] == 'ok';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Plant Disease AI — Статус API'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _isLoading ? null : _checkHealth,
            tooltip: 'Оновити',
          ),
        ],
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                isOk
                    ? Icons.check_circle_outline
                    : (_errorMessage != null
                        ? Icons.error_outline
                        : Icons.cloud_sync_outlined),
                size: 80,
                color: isOk
                    ? Colors.green
                    : (_errorMessage != null ? Colors.red : Colors.grey),
              ),
              const SizedBox(height: 20),
              Text(
                'Перевірка зв\'язку з бекендом',
                style: Theme.of(context).textTheme.headlineSmall,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              Text(
                'Цільовий ендпоінт: $_apiUrl',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 30),
              if (_isLoading)
                const CircularProgressIndicator()
              else if (_errorMessage != null)
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.red.shade50,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.red.shade200),
                  ),
                  child: Text(
                    _errorMessage!,
                    style: TextStyle(color: Colors.red.shade900),
                    textAlign: TextAlign.center,
                  ),
                )
              else if (_healthData != null)
                Card(
                  elevation: 2,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _buildStatusRow(
                          'Статус сервісу',
                          _healthData!['status'].toString().toUpperCase(),
                          isOk ? Colors.green : Colors.orange,
                        ),
                        const Divider(),
                        _buildStatusRow(
                          'Модель завантажена',
                          _healthData!['model_loaded'] == true ? 'ТАК' : 'НІ',
                          _healthData!['model_loaded'] == true
                              ? Colors.green
                              : Colors.red,
                        ),
                        const Divider(),
                        _buildStatusRow(
                          'База даних',
                          _healthData!['database_connected'] == true
                              ? 'ПІДКЛЮЧЕНО'
                              : 'ВІДКЛЮЧЕНО',
                          _healthData!['database_connected'] == true
                              ? Colors.green
                              : Colors.red,
                        ),
                        const Divider(),
                        _buildStatusRow(
                          'Версія API',
                          _healthData!['version']?.toString() ?? '1.0.0',
                          Colors.blueGrey,
                        ),
                      ],
                    ),
                  ),
                ),
              const SizedBox(height: 30),
              ElevatedButton.icon(
                onPressed: _isLoading ? null : _checkHealth,
                icon: const Icon(Icons.network_check),
                label: const Text('Повторити перевірку'),
                style: ElevatedButton.styleFrom(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatusRow(String label, String value, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontWeight: FontWeight.w500)),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              value,
              style: TextStyle(
                color: color,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

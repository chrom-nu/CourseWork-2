import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/main.dart';

void main() {
  testWidgets('Health check screen smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const PlantDiseaseApp());

    // Verify app title exists
    expect(find.text('Перевірка зв\'язку з бекендом'), findsOneWidget);
    expect(find.text('Повторити перевірку'), findsOneWidget);
  });
}

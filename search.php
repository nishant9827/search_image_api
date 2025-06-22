<?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $image = $_FILES['image'];
    $path = 'uploads/' . basename($image['name']);
    move_uploaded_file($image['tmp_name'], $path);

    $cfile = new CURLFile($path);
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, "http://localhost:5000/search"); // Update if hosted elsewhere
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, ['image' => $cfile]);
    $response = curl_exec($ch);
    curl_close($ch);

    $product_ids = json_decode($response);

    $pdo = new PDO("mysql:host=localhost;dbname=u903684843_dabramart", "u903684843_dabramart", "jY[r0g@8");
    $placeholders = implode(',', array_fill(0, count($product_ids), '?'));
    $stmt = $pdo->prepare("SELECT * FROM product_items WHERE product_id IN ($placeholders)");
    $stmt->execute($product_ids);
    $results = $stmt->fetchAll();

    foreach ($results as $item) {
        echo "<div>
                <img src='https://dabramart.in/admin_panel/{$item['product_image']}' width='120'><br>
                <strong>{$item['product_name']}</strong>
              </div><hr>";
    }
}
?>

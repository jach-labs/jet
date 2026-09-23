// Clear only the cache created by the retired browser-inference demo.
// This page never downloads or initializes model weights.
if ("caches" in window) {
  caches.delete("jet-8a97cfea2df622bb03f5dc9b02567e21abd2551c").catch(() => {});
}

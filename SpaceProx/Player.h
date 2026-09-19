#pragma once

namespace SpaceProx
{
class Player
{
public:
    Player();

    virtual void Update(float deltaSeconds);
    virtual void TakeDamage(int amount);
    virtual int GetHealth();
    virtual bool IsInvincible();
    virtual float GetSpeed();  
    virtual bool IsEnemyVisible(int enemyIndex); 
    virtual int GetScore();
    virtual int GetAmmo();
    virtual bool ConsumeAmmo();   
    int GetDamageTaken(int amount);
    int health;
    int score;
    int ammo;
    float x, y;
    float aimX, aimY;
    float hitFlash;
    float fireCooldown;
    float reloadTime;
};

// Defined in a separate translation unit, keeping the concrete object opaque
// to the renderer and preserving virtual dispatch in optimized builds.
__declspec(noinline) Player* CreatePlayer();
} // namespace SpaceProx
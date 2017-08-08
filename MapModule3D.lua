local MapModule3D, parent = torch.class('nn.MapModule3D', 'nn.Sequential')
function MapModule3D:__init(module)
  -- expects 3d input
  parent.__init(self)

  self:add(nn.SplitTable(1))
      :add(nn.MapTable():add(nn.Sequential():add(module):add(nn.Unsqueeze(1))))
      :add(nn.JoinTable(1))
end